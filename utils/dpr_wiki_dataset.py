# coding=utf-8

"""Wikipedia dataset from DPR code for ORQA."""

import sys
import csv
from abc import ABC

import numpy as np
from torch.utils.data import Dataset

from utils import print_rank_0

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
                                           tokens_encode_func)
    return dataset


def get_open_retrieval_batch(data_iterator):
    if data_iterator is None:
        data = None
    else:
        data = next(data_iterator)

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

    def __init__(self, task_name, dataset_name, datapath, tokens_encode_func):
        # Store inputs.
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.enc_func = tokens_encode_func
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
        sample = build_sample(row['doc_id'], row)
        return sample

    @staticmethod
    def process_samples_from_single_path(filename):
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
