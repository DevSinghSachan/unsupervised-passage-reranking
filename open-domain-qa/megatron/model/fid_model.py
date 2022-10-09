import warnings
import numpy as np
import torch
from megatron import get_args, print_rank_0
from megatron.checkpointing import load_t5_checkpoint
from megatron.model import T5Model
from megatron.module import MegatronModule
from megatron import get_t5_tokenizer
from megatron.tokenizer.tokenizer import vocab_size_with_padding
from megatron.data.mask_creation_utils import make_attention_mask_3d, make_history_mask_3d
from tools.inverted_title_index import WikiTitleDocMap
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.data.orqa_wiki_dataset import build_tokens_types_paddings_from_ids as context_bert_format


def flatten(ids, types):
    _, _, max_seq_len = ids.shape
    # B x K x T -> B K x T
    ids = ids.reshape(-1, max_seq_len)
    # B x K x T -> B K x T
    types = types.reshape(-1, max_seq_len)
    return ids, types


class FiDModel(MegatronModule):
    def __init__(self, evidence_indexer):
        super(FiDModel, self).__init__()
        args = get_args()
        self.topk = args.topk_retrievals

        print_rank_0('building Reader for FiD ...')
        t5_tokenizer = get_t5_tokenizer()
        t5_vocab_size = vocab_size_with_padding(t5_tokenizer.vocab_size,
                                                args)

        self.language_model = T5Model(num_tokentypes=2,
                                      parallel_output=True,
                                      vocab_size=t5_vocab_size)
        self._language_model_key = 'encoder/t5_model'

        self.evidence_indexer = evidence_indexer
        self.t5_tokenizer = get_t5_tokenizer()


    def forward(self, query_uid, query_ids_bert, query_types, query_mask_bert, ctx_ids,
                query_ids_t5, query_ids_t5_len, dec_ids,
                all_query_context_hidden_states=None, all_query_context_ids_unflat=None):

        args = get_args()
        bsize, max_seq_len = query_ids_bert.shape

        if all_query_context_hidden_states is None:

            topk_evidence_data = self.evidence_indexer.get_topk(ctx_ids)

            with torch.no_grad():
                output = postprocess(query_uid,
                                     query_ids_t5,
                                     query_ids_t5_len,
                                     topk_evidence_data)
                all_query_extended_context_ids, query_one_context_ids = output

            # MASK Handling of topk-augmented encoders
            all_query_context_mask = make_attention_mask_3d(all_query_extended_context_ids, all_query_extended_context_ids)
            all_query_context_mask = all_query_context_mask < 0.5

            # [batch_size x k, args.seq_length_dec, hidden_size]
            all_query_context_hidden_states = self.language_model(encoder_input_ids=all_query_extended_context_ids,
                                                                  decoder_input_ids=dec_ids,
                                                                  encoder_attn_mask=all_query_context_mask,
                                                                  decoder_attn_mask=None,
                                                                  encoder_decoder_attn_mask=None,
                                                                  output_enc_hidden=True)

            # Reshape the query context hidden states
            all_query_context_hidden_states = all_query_context_hidden_states.reshape(bsize,
                                                                                      args.topk_retrievals * args.seq_length,
                                                                                      args.hidden_size)

            all_query_context_ids_unflat = all_query_extended_context_ids.reshape(bsize,
                                                                                  args.topk_retrievals * args.seq_length)

        enc_dec_mask = make_attention_mask_3d(dec_ids, all_query_context_ids_unflat)
        enc_dec_mask = enc_dec_mask < 0.5

        dec_mask = make_attention_mask_3d(dec_ids, dec_ids)
        dec_mask = dec_mask * make_history_mask_3d(dec_ids)
        dec_mask = dec_mask < 0.5

        # Calculate the LM logits
        # When we already have the encoder's hidden states, then all_query_context_ids_unflat is not important
        # Truncating the max sequence length to limit to max-sequence-length
        temp_ids = all_query_context_ids_unflat[:, :args.seq_length]

        lm_logits, _ = self.language_model(temp_ids,
                                           dec_ids,
                                           encoder_attn_mask=None,
                                           decoder_attn_mask=dec_mask,
                                           encoder_decoder_attn_mask=enc_dec_mask,
                                           enc_hidden_states=all_query_context_hidden_states)

        if self.training:
            return lm_logits
        else:
            return lm_logits, all_query_context_hidden_states, all_query_context_ids_unflat


    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads, add an extra key."""
        state_dict_ = dict()
        state_dict_[self._language_model_key] = self.language_model.state_dict_for_save_checkpoint(destination,
                                                                                                   prefix,
                                                                                                   keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dicts of each of the models"""
        self.language_model.load_state_dict(state_dict[self._language_model_key], strict)

    def init_state_dict_from_dpr_and_t5(self):
        """Initialize the state from pre-trained DPR model and pre-trained T5 mode on iteration zero of pretraining"""
        args = get_args()

        if args.pretrained_t5_load is None:
            warnings.warn("Pretrained Checkpoints are not found. Initializing from random weights")
            return

        print("Initializing reader model from pretrained T5", flush=True)
        load_t5_checkpoint(self.language_model,
                           custom_load_path=args.pretrained_t5_load)


def postprocess(query_uid, query_ids_t5, query_ids_t5_len, topk_evidence_data):
    args = get_args()
    query_uid = query_uid.tolist()
    t5_tokenizer = get_t5_tokenizer()

    all_context_ids, all_context_types = [], []
    all_query_extended_context_ids = []
    all_query_one_context_ids = []

    for qid, query_t5, query_t5_len, (topkids, text_list) in zip(query_uid, query_ids_t5, query_ids_t5_len, topk_evidence_data):
        k = 0
        query_t5 = query_t5.tolist()[:query_t5_len]
        context_ids_list, context_types_list = [], []
        for eid, (context_doc_list, main_doc_idx, title_ids) in zip(topkids, text_list):
            # We should ignore the evidence from which query originates
            if qid != eid and k < args.topk_retrievals:
                k += 1
                # Except for the masked tokens from extra-vocab-ids, BERT tokenizer and T5 tokenizer output the same encodings
                context_ids = context_doc_list[main_doc_idx]

                ids, types, pad_mask = context_bert_format(title_ids + [t5_tokenizer.sep] + context_ids,
                                                           args.seq_length_ret,
                                                           t5_tokenizer.cls,
                                                           t5_tokenizer.sep,
                                                           t5_tokenizer.pad)
                context_ids_list.append(ids)
                context_types_list.append(types)

                # Jointly encode the query and ...extended context tokens...
                query_context_ids = query_extended_context_t5_format(query_t5,
                                                                     title_ids,
                                                                     context_doc_list,
                                                                     main_doc_idx,
                                                                     args.seq_length,
                                                                     t5_tokenizer.sep,
                                                                     t5_tokenizer.pad)
                all_query_extended_context_ids.append(query_context_ids)

                # Jointly encode the query and ...single context tokens...
                query_one_context_ids = query_single_context_t5_format(query_t5,
                                                                       title_ids,
                                                                       context_ids,
                                                                       args.seq_length,
                                                                       t5_tokenizer.sep,
                                                                       t5_tokenizer.pad)
                all_query_one_context_ids.append(query_one_context_ids)

        all_context_ids.append(np.array(context_ids_list))
        all_context_types.append(np.array(context_types_list))

    return torch.cuda.LongTensor(all_query_extended_context_ids), \
           torch.cuda.LongTensor(all_query_one_context_ids)


def query_extended_context_t5_format(query_ids, title_ids, context_doc_list, main_doc_idx, max_seq_length, sep_id, pad_id):
    enc_ids = query_ids + title_ids + [sep_id]

    def prepare_context_ids(maxlen):
        context_ids = context_doc_list[main_doc_idx]

        if len(context_ids) > maxlen or len(context_doc_list) == 1:
            context_ids = context_ids[0: maxlen]
            return context_ids
        else:
            extra_len = maxlen - len(context_ids)
            if main_doc_idx == 0:
                extra_context_ids = []
                for item in context_doc_list[1:]:
                    extra_context_ids.extend(item)
                if len(extra_context_ids) > extra_len:
                    extra_context_ids = extra_context_ids[0: extra_len]
                context_ids = context_ids + extra_context_ids
                return context_ids
            elif main_doc_idx == -1:
                extra_context_ids = []
                for item in context_doc_list[:-1]:
                    extra_context_ids.extend(item)
                if len(extra_context_ids) > extra_len:
                    offset = len(extra_context_ids) - extra_len + 1
                    extra_context_ids = extra_context_ids[offset:]
                context_ids = extra_context_ids + context_ids
                return context_ids
            else:  # for condition main_doc_idx=1
                left_extra_context_ids = context_doc_list[0]
                if len(left_extra_context_ids) > extra_len:
                    offset = len(left_extra_context_ids) - extra_len + 1
                    left_extra_context_ids = left_extra_context_ids[offset:]
                    context_ids = left_extra_context_ids + context_ids
                    return context_ids
                context_ids = left_extra_context_ids + context_ids
                if len(context_doc_list) == 3:
                    right_extra_context_ids = context_doc_list[2]
                    len_remaining = extra_len - len(left_extra_context_ids)
                    if len(right_extra_context_ids) > len_remaining:
                        right_extra_context_ids = right_extra_context_ids[:len_remaining]
                    context_ids = context_ids + right_extra_context_ids
                return context_ids

    remaining_len = max(0, max_seq_length - len(enc_ids) - 1)
    extended_context_ids = prepare_context_ids(remaining_len)
    enc_ids.extend(extended_context_ids)
    enc_ids.append(sep_id)

    padding_length = max_seq_length - len(enc_ids)
    if padding_length > 0:
        enc_ids.extend([pad_id] * padding_length)

    return enc_ids


def query_single_context_t5_format(query_ids, title_ids, context_ids, max_seq_length, sep_id, pad_id):
    enc_ids = []
    src_ids = query_ids + title_ids + [sep_id] + context_ids
    enc_ids.extend(src_ids)

    if len(enc_ids) > max_seq_length - 1:
        enc_ids = enc_ids[0: max_seq_length - 1]

    enc_ids.append(sep_id)

    padding_length = max_seq_length - len(enc_ids)
    if padding_length > 0:
        enc_ids.extend([pad_id] * padding_length)

    return enc_ids


class EvidenceDocsIndexer(object):
    def __init__(self):
        args = get_args()
        self.topk = args.topk_retrievals

        self.passages_map = make_indexed_dataset(args.indexed_evidence_data_path,
                                                 impl=args.data_impl,
                                                 skip_warmup=(not args.mmap_warmup))

        self.title_map = make_indexed_dataset(args.indexed_title_data_path,
                                              impl=args.data_impl,
                                              skip_warmup=(not args.mmap_warmup))

        self.wikititledocmap = WikiTitleDocMap(args.evidence_data_path)


    def get_topk(self, topkindex):
        topk_data = []
        for topkarray in topkindex:
            # The idx contains passage text and title
            topkarray = topkarray.tolist()
            text_list = []
            for idx in topkarray:
                doc_idxs, main_doc_idx = self.wikititledocmap.get_neighbour_paragraphs(idx)
                doc_list = [self.passages_map[doc_id-1].tolist() for doc_id in doc_idxs]
                text_list.append((doc_list,
                                  main_doc_idx,
                                  self.title_map[idx-1].tolist()))
            topk_data.append((topkarray, text_list))

        return topk_data
