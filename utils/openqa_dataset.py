from abc import ABC
import json
from collections import OrderedDict
import random
import torch
from torch.utils.data import Dataset, DataLoader
from utils import print_rank_0
from utils.data_utils import DistributedBatchSampler


def get_openqa_dataset(task_name, dataset_path, sample_rate=1.0):
    dataset = OpenQADataset(task_name,
                            "open-domain retrieval",
                            dataset_path,
                            sample_rate)
    return dataset



class OpenQADataset(ABC, Dataset):
    def __init__(self, task_name, dataset_name, filepath, sample_rate):
        self.task_name = task_name
        self.dataset_name = dataset_name
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name,
                                                             self.dataset_name))
        self.samples = self.load_dataset(filepath)

        if sample_rate < 1:  # subsample
            k = int(len(self.samples) * sample_rate)
            self.samples = random.sample(self.samples, k)

        print_rank_0('  >> total number of samples: {}'.format(len(self.samples)))

        if "trivia" in filepath or 'webq' in filepath or 'entity-questions' in filepath \
                or "BEIR" in filepath or "squad" in filepath:
            self.ques_punc = ""
        elif "nq" in filepath or "efficientqa" in filepath:
            self.ques_punc = "?"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        # These [CLS] and [SEP] tokens exist due to BERT tokenization, so we need to remove them
        if "[CLS]" and "[SEP]" in row['question']:
            row['question'] = " ".join(row['question'].split()[1:-1])

        if self.task_name == "reranking":
            decoder_prompt = "Question: {}{}".format(row['question'], self.ques_punc)
        else:
            raise AssertionError("invalid --task-name argument {}".format(self.task_name))

        encoder_contexts = None
        if 'ctxs' in row:
            encoder_contexts = row['ctxs']
        elif 'contexts' in row:
            encoder_contexts = row['contexts']

        answers = row['answers']

        sample = {'id': idx,
                  'encoder_ids': encoder_contexts,
                  'decoder_ids': decoder_prompt,
                  'question': row['question'],
                  'answers': answers}
        return sample

    @staticmethod
    def load_dataset(filepath):
        with open(filepath) as fp:
            data = json.load(fp)

        # condition for interfacing with pyserineni BM25 outputs
        if isinstance(data, dict):
            return list(data.values())
        else:
            return data


def get_one_epoch_dataloader(dataset, args, batch_size=None):
    """Specifically one epoch to be used in an indexing job."""
    # args = get_args()

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    if batch_size is None:
        batch_size = args.batch_size

    global_batch_size = batch_size * world_size
    num_workers = args.num_workers

    sampler = torch.utils.data.SequentialSampler(dataset)

    # importantly, drop_last must be False to get all the data.
    batch_sampler = DistributedBatchSampler(sampler,
                                            batch_size=global_batch_size,
                                            drop_last=False,
                                            rank=rank,
                                            world_size=world_size)

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = CustomDataLoader(dataset,
                                   batch_sampler=batch_sampler,
                                   num_workers=num_workers,
                                   pin_memory=True)
    return data_loader


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, eval=False, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn
        self.eval = eval
        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        # generate batch
        batch_size = len(batch_data)
        if batch_size == 0:
            raise StopIteration
        tensorized = OrderedDict()
        for d in batch_data:
            for k, v in d.items():
                tensorized.setdefault(k, []).append(v)
        assert len(tensorized) == 5

        return tensorized
