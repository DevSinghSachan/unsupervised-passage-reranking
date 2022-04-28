# coding=utf-8

"""Wikipedia dataset from DPR code for ORQA."""

import sys
import csv
import time
from abc import ABC

import numpy as np
import torch
from torch.utils.data import Dataset

from upr import print_rank_0

# https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


def get_open_retrieval_wiki_dataset(args, tokens_encode_func):

    dataset = OpenRetrievalEvidenceDataset('2018 Wikipedia from DPR codebase',
                                           'evidence',
                                           args.evidence_data_path,
                                           tokens_encode_func,
                                           args.seq_length_ret)
    return dataset


def get_open_retrieval_batch(data_iterator):
    # Items and their type.
    keys = ['row_id', 'context', 'context_mask', 'context_types', 'context_pad_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is None:
        data = None
    else:
        data = next(data_iterator)
    # data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    row_id = data['row_id'].long().cuda()
    context = data['context'].long().cuda()

    return row_id, context


def build_sample(row_id, context_ids):
    """Convert to numpy and return a sample consumed by the batch producer."""

    sample = ({
        'row_id': row_id,
        'context': np.array(context_ids, dtype=np.int64)
    })
    return sample


class OpenRetrievalEvidenceDataset(ABC, Dataset):
    """Open Retrieval Evidence dataset class."""

    def __init__(self, task_name, dataset_name, datapath, tokens_encode_func, max_seq_length):
        # Store inputs.
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.enc_func = tokens_encode_func
        self.max_seq_length = max_seq_length
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name,
                                                             self.dataset_name))
        # Process the files.
        print_rank_0(datapath)
        self.samples, self.id2text = self.process_samples_from_single_path(datapath)

        print_rank_0('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        context_ids = self.enc_func(row['title'], row['text'])
        context_ids = build_tokens_paddings_from_ids(context_ids.tolist(),
                                                     max_seq_length=self.max_seq_length)

        sample = build_sample(row['doc_id'],
                              context_ids)
        return sample

    @staticmethod
    def process_samples_from_single_path(filename):
        start_time = time.time()

        print_rank_0(' > Processing {} ...'.format(filename))
        total = 0

        rows = []
        id2text = {}

        with open(filename) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            next(reader, None)  # skip the headers
            for row in reader:
                # file format: doc_id, doc_text, title
                doc_id = int(row[0])
                text = row[1]
                title = row[2]

                rows.append({'doc_id': doc_id,
                             'text': text,
                             'title': title})

                assert doc_id not in id2text
                id2text[doc_id] = (text, title)

                total += 1
                if total % 100000 == 0:
                    print_rank_0('  > processed {} rows so far ...'.format(total))

        print_rank_0(' >> processed {} samples.'.format(len(rows)))

        return rows, id2text



def build_tokens_paddings_from_ids(text_ids, max_seq_length, bos_id=0, eos_id=2, pad_id=1):
    """Build token types and paddings, trim if needed, and pad if needed."""
    enc_ids = []

    # A.
    len_src = len(text_ids)
    enc_ids.extend(text_ids)

    # Cap the size.
    if len(enc_ids) > max_seq_length - 1:
        enc_ids = enc_ids[0: max_seq_length - 1]
        enc_ids.append(eos_id)

    # Padding.
    padding_length = max_seq_length - len(enc_ids)
    if padding_length > 0:
        enc_ids.extend([pad_id] * padding_length)

    return enc_ids
