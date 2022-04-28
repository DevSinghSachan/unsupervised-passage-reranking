import torch
import torch.distributed as dist
import random
import numpy
import json
import time
import argparse
import os
import shutil
from upr import print_rank_0
from upr.nq_dataset import get_nq_open_dataset, get_one_epoch_dataloader
from upr.initialize import initialize_distributed
from transformers import T5Tokenizer, T5ForConditionalGeneration
from upr.orqa_wiki_dataset import get_open_retrieval_wiki_dataset


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


class Reranking():
    def __init__(self, args, call_load_attributes_func=True):
        self.model = None
        self.dataloader = None
        self.dataset = None
        self.evidence_dataset = None

        self.args = args

        self.log_interval = args.log_interval
        self.batch_size = args.batch_size

        if call_load_attributes_func:
            self.load_attributes()
        self.is_main_builder = dist.get_rank() == 0
        self.num_total_builders = dist.get_world_size()

        self.temp_dir_name = os.path.join(args.output_path, '_tmp_reranker_{}'.format(os.getenv("SLURM_JOBID")))

    def load_attributes(self, custom_load_path=None, key_list=None):
        print_rank_0("Loading {} weights".format(self.args.hf_model_name))
        self.tokenizer = T5Tokenizer.from_pretrained(self.args.hf_model_name)

        if self.args.use_fp16:
            self.model = T5ForConditionalGeneration.from_pretrained(self.args.hf_model_name, torch_dtype=torch.float16)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(self.args.hf_model_name)
            # parallelize(self.model, num_gpus=8, fp16=False, verbose='detail')

        if self.args.use_gpu:
            self.model = self.model.cuda()

        print_rank_0("Loaded {} weights".format(self.args.hf_model_name))

        self.model.eval()  # disable dropout (or leave in train mode to finetune)

        self.evidence_dataset = get_open_retrieval_wiki_dataset(args=self.args,
                                                                tokens_encode_func=None)

        self.dataset = get_nq_open_dataset(self.args.task_name,
                                           self.args.retriever_output_path,
                                           sample_rate=self.args.sample_rate)

        self.dataloader = iter(get_one_epoch_dataloader(self.dataset,
                                                        self.args,
                                                        self.batch_size))
        self.iteration = self.total_processed = 0

    def track_and_report_progress(self, batch_size):
        """Utility function for tracking progress"""
        self.iteration += 1
        self.total_processed += batch_size * self.num_total_builders
        if self.is_main_builder and self.iteration % self.log_interval == 0:
            print('Batch {:10d} | Total {:10d}'.format(self.iteration, self.total_processed), flush=True)

    def do_inference(self):
        """Goes through one epoch of the dataloader and adds all data to this instance's BlockData.

        The copy of BlockData is saved as a shard, which when run in a distributed setting will be
        consolidated by the rank 0 process and saved as a final pickled BlockData.
        """

        reranked_answers_list = []
        original_answers_list = []
        reranked_data = []

        start_time = time.time()

        while True:
            try:
                # batch also has query_tokens and query_pad_data
                batch = next(self.dataloader)
            except (StopIteration, IndexError):
                break

            assert len(batch['id']) == 1, "Currently, we are doing inference with batch size 1"

            all_contexts = batch['encoder_ids'][0][:self.args.topk_contexts]

            all_ids = []
            has_answer_list = []

            for i, context in enumerate(all_contexts):
                # ids = "Context: {} {}".format(context.get('title'), context.get('text'))
                # ids = "Generate a question based on this passage: {} {}".format(context.get('title'), context.get('text'))
                text, title = self.evidence_dataset.id2text[int(context.get("id"))]
                ids = "{} {} {}. {}".format(self.args.verbalizer_head, title, text, self.args.verbalizer)
                all_ids.append(ids)
                has_answer_list.append(context.get('has_answer'))

            input_encoding = self.tokenizer(all_ids,
                                            padding='longest',
                                            max_length=512,
                                            truncation=True,
                                            return_tensors='pt')

            if self.args.use_gpu:
                context_tensor, attention_mask = input_encoding.input_ids.cuda(), input_encoding.attention_mask.cuda()
            else:
                context_tensor, attention_mask = input_encoding.input_ids, input_encoding.attention_mask

            decoder_prefix = batch['decoder_ids']
            target_encoding = self.tokenizer(decoder_prefix,
                                             max_length=128,
                                             truncation=True,
                                             return_tensors='pt')

            if self.args.use_gpu:
                decoder_prefix_tensor = target_encoding.input_ids.cuda()
            else:
                decoder_prefix_tensor = target_encoding.input_ids

            decoder_prefix_tensor = torch.repeat_interleave(decoder_prefix_tensor,
                                                            len(context_tensor),
                                                            dim=0)

            sharded_nll_list = []

            for i in range(0, len(context_tensor), self.args.shard_size):
                encoder_tensor_view = context_tensor[i: i + self.args.shard_size]
                attention_mask_view = attention_mask[i: i + self.args.shard_size]
                decoder_tensor_view = decoder_prefix_tensor[i: i + self.args.shard_size]
                with torch.no_grad():
                    logits = self.model(input_ids=encoder_tensor_view,
                                        attention_mask=attention_mask_view,
                                        labels=decoder_tensor_view).logits

                log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
                nll = -log_softmax.gather(2, decoder_tensor_view.unsqueeze(2)).squeeze(2)

                avg_nll = torch.sum(nll, dim=1)
                sharded_nll_list.append(avg_nll)

            topk_scores, indexes = torch.topk(-torch.cat(sharded_nll_list), k=len(context_tensor))
            ranked_answers = torch.BoolTensor(has_answer_list)[indexes]

            # Save the essential information to be used for saving the re-ranked information component.
            original_answers_list.append(has_answer_list)
            reranked_answers_list.append(ranked_answers.tolist())

            reordered_context = [all_contexts[i] for i in indexes]

            for i, ctx in enumerate(reordered_context):
                ctx['score'] = topk_scores[i].item()

            item = {"question": batch['question'][0],
                    "answers": batch['answers'][0],
                    "ctxs": reordered_context[:self.args.report_topk_accuracies[-1]]}
            reranked_data.append(item)

            self.track_and_report_progress(batch_size=len(batch['id']))

        end_time = time.time()
        time_taken = (end_time - start_time) / len(reranked_data)
        torch.distributed.barrier()

        print_rank_0("Time taken: {} seconds".format(time_taken))

        self.compute_topk_recall(original_answers_list, string_prefix="Original Ranking")
        self.compute_topk_recall(reranked_answers_list, string_prefix="Re-Ranking")

        if self.args.trec_eval:
            self.trec_eval(reranked_data)
            self.recall_cap(reranked_data)

        if self.args.merge_shards_and_save:
            self.save_shard(reranked_data)

        del self.model
        # This process signals to finalize its shard and then synchronize with the other processes
        torch.distributed.barrier()

        if self.args.merge_shards_and_save:
            # rank 0 process builds the final copy
            if self.is_main_builder:
                self.merge_shards_and_save()
            # complete building the final copy
            torch.distributed.barrier()

    def trec_eval(self, reranked_data):
        ndcg = {}
        recall = {}

        for k in self.args.report_topk_accuracies:
            ndcg[f"NDCG@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0

        qrels = {}
        results = {}
        for i, item in enumerate(reranked_data):
            qrels[str(i)] = {str(k): v for k, v in item["answers"].items()}
            results[str(i)] = {str(k["id"]): k["score"] for k in item["ctxs"]}

        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in self.args.report_topk_accuracies])
        recall_string = "recall." + ",".join([str(k) for k in self.args.report_topk_accuracies])

        import pytrec_eval
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string, recall_string})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in self.args.report_topk_accuracies:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

        ndcg_tensor = torch.FloatTensor([ndcg[f"NDCG@{k}"] for k in self.args.report_topk_accuracies]).cuda()
        torch.distributed.all_reduce(ndcg_tensor, torch.distributed.ReduceOp.SUM)

        recall_tensor = torch.FloatTensor([recall[f"Recall@{k}"] for k in self.args.report_topk_accuracies]).cuda()
        torch.distributed.all_reduce(recall_tensor, torch.distributed.ReduceOp.SUM)

        n_queries = torch.FloatTensor([len(scores)]).cuda()
        torch.distributed.all_reduce(n_queries, torch.distributed.ReduceOp.SUM)

        if torch.distributed.get_rank() == 0:
            ndcg_tensor = ndcg_tensor / n_queries
            recall_tensor = recall_tensor / n_queries

            for i, k in enumerate(self.args.report_topk_accuracies):
                print_rank_0("NDCG@{}: {:.4f}".format(k, ndcg_tensor[i] * 100))
            print_rank_0("\n")

            for i, k in enumerate(self.args.report_topk_accuracies):
                print_rank_0("Recall@{}: {:.4f}".format(k, recall_tensor[i] * 100))

    def recall_cap(self, reranked_data):
        capped_recall = {}
        for k in self.args.report_topk_accuracies:
            capped_recall[f"R_cap@{k}"] = 0.0

        k_max = max(self.args.report_topk_accuracies)

        qrels = {}
        results = {}
        for i, item in enumerate(reranked_data):
            qrels[str(i)] = {str(k): v for k, v in item["answers"].items()}
            results[str(i)] = {str(k["id"]): k["score"] for k in item["ctxs"]}

        for query_id, doc_scores in results.items():
            top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
            query_relevant_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
            for k in self.args.report_topk_accuracies:
                retrieved_docs = [row[0] for row in top_hits[0:k] if qrels[query_id].get(row[0], 0) > 0]
                denominator = min(len(query_relevant_docs), k)
                capped_recall[f"R_cap@{k}"] += (len(retrieved_docs) / denominator)

        capped_recall_tensor = torch.FloatTensor([capped_recall[f"R_cap@{k}"] for k in self.args.report_topk_accuracies]).cuda()
        torch.distributed.all_reduce(capped_recall_tensor, torch.distributed.ReduceOp.SUM)

        n_queries = torch.FloatTensor([len(results)]).cuda()
        torch.distributed.all_reduce(n_queries, torch.distributed.ReduceOp.SUM)

        if torch.distributed.get_rank() == 0:
            capped_recall_tensor = capped_recall_tensor / n_queries

            print_rank_0("\n")
            for i, k in enumerate(self.args.report_topk_accuracies):
                print_rank_0("Capped-Recall@{}: {:.4f}".format(k, capped_recall_tensor[i] * 100))
            print_rank_0("\n")

        return capped_recall

    @staticmethod
    def calculate_topk_hits(scores, max_k):
        top_k_hits = [0] * max_k
        for question_hits in scores:
            best_hit = next((i for i, x in enumerate(question_hits[:max_k]) if x), None)
            if best_hit is not None:
                top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
        return top_k_hits

    def compute_topk_recall(self, answers_list, string_prefix):
        topk_hits = self.calculate_topk_hits(answers_list, max_k=self.args.report_topk_accuracies[-1])

        topk_hits = torch.FloatTensor(topk_hits).cuda()
        n_docs = torch.FloatTensor([len(answers_list)]).cuda()
        torch.distributed.all_reduce(topk_hits, torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(n_docs, torch.distributed.ReduceOp.SUM)

        if torch.distributed.get_rank() == 0:
            topk_hits = topk_hits / n_docs
            print(string_prefix)
            for i in self.args.report_topk_accuracies:
                print_rank_0("top-{}: {:.2f}".format(i, topk_hits[i - 1] * 100))
            print("\n")

    def save_shard(self, data):
        """
        Save the block data that was created this in this process
        """
        if not os.path.isdir(self.temp_dir_name):
            os.makedirs(self.temp_dir_name, exist_ok=True)

        outpath = os.path.join(self.temp_dir_name, "rank{}.json".format(dist.get_rank()))
        with open(outpath, "w") as writer:
            writer.write(json.dumps(data, indent=4) + "\n")


    def merge_shards_and_save(self):
        """Combine all the shards made using self.save_shard()"""
        shard_names = os.listdir(self.temp_dir_name)
        all_data = []

        for fname in os.listdir(self.temp_dir_name):
            shard_size = 0
            old_size = len(all_data)
            fpath = '{}/{}'.format(self.temp_dir_name, fname)
            with open(fpath, 'r') as f:
                data = json.load(f)
                shard_size = len(data)
                all_data.extend(data)

            assert len(all_data) == old_size + shard_size
            os.remove(fpath)

        # save the consolidated shards
        outpath = os.path.join(self.args.output_path, "{}.json".format(self.args.special_suffix))

        with open(outpath, 'w') as writer:
            writer.write(json.dumps(all_data, indent=4) + "\n")

        print("Finished merging {} shards for a total of {} embeds".format(
            len(shard_names), len(all_data)), flush=True)

        # make sure that every single piece of data was embedded
        assert len(all_data) == len(self.dataset)

        shutil.rmtree(self.temp_dir_name, ignore_errors=True)


def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title='output data')

    group.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher.')

    group.add_argument('--main-port', type=int, default=29500,
                       help='Main port number.')

    group.add_argument('--special-suffix', type=str, default="",
                       help='special suffix extension for saving merged file')

    group.add_argument('--retriever-output-path', type=str, default="/checkpoint/dsachan/retriever-outputs/nq-dev.json",
                       help='Path of the Top-K outputs from retriever (.json file)')

    group.add_argument('--topk-contexts', type=int, default=100,
                       help='number of topk context to select')

    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')

    group.add_argument('--batch-size', type=int, default=1)

    group.add_argument('--shard-size', type=int, default=50)

    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")

    group.add_argument('--output-path', type=str, default="/checkpoint/dsachan/marge-reranking/",
                       help='Where to save inference results')

    group.add_argument('--task-name', type=str, default="reranking",
                       help='Name of the task.')

    group.add_argument('--hf-model-name', type=str, default="t5-large",
                       help='Name of the HF model.')

    group.add_argument('--interactive-node', action='store_true',
                       help='If the node is interactive or not')

    group.add_argument('--use-gpu', action='store_true',
                       help='Use GPU or not')

    group.add_argument('--use-fp16', action='store_true',
                       help='Use FP16 or not')

    group.add_argument('--use-nccl-reduce', action='store_true',
                       help='Use nccl reduce or not')

    group.add_argument('--merge-shards-and-save', action='store_true',
                       help='whether to merge individual data shards or not for reranking')

    group.add_argument('--trec-eval', action='store_true',
                       help='Whether to use trec evaluation tools')

    group.add_argument('--sample-rate', type=float, default=1.,
                       help="Sample rate for the number of examples.")

    group.add_argument('--random-seed', type=int, default=1234,
                       help="Random seed.")

    group.add_argument('--evidence-data-path', type=str, default=None,
                       help='Path to Wikipedia Evidence frm DPR paper')

    group.add_argument('--verbalizer', type=str, default="Please write a question based on this passage.",
                       help='Prompt string for generating the target tokens')

    group.add_argument('--verbalizer-head', type=str, default="Passage: ",
                       help='The string token used to represent encoder input')

    group.add_argument('--report-topk-accuracies', nargs='+', type=int, default=[1, 5, 10, 20, 50, 100],
                       help="Which top-k accuracies to report (e.g. '1 5 20')")

    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    # Distributed args.
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    # Just putting a dummy value here
    args.seq_length_ret = 256

    return args


def main():
    args = get_args()
    set_random_seed(args.random_seed)
    initialize_distributed(args)

    reranker = Reranking(args)
    reranker.do_inference()


if __name__ == "__main__":
    main()
