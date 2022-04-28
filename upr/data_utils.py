import torch
from torch.utils import data


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

    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)


class DistributedBatchSampler(data.sampler.BatchSampler):
    """Similar to normal implementation of distributed sampler, except
    implementation is at the batch sampler level, instead of just the
    sampler level. This allows wrapping of arbitrary data samplers
    (sequential, random, WeightedRandomSampler, etc.) with this batch
    sampler.

    The `interleave` argument specifies how to distribute a batch. A value
    of True combined with the above random sampler is equivalent to pytorch's
    torch.utils.data.distributed.DistributedSampler.

    For the following batch [0,1,2,3,4,5,6,7] and data parallelism of 2
    specifying True will result in the following samples for each gpu:
        GPU0: [0,2,4,6] GPU1: [1,3,5,7]
    specifying False will result in the following samples:
        GPU0: [0,1,2,3] GPU1: [4,5,6,7]"""

    def __init__(self, sampler, batch_size, drop_last, rank=-1,
                 world_size=2, wrap_last=False, interleave=False):
        super(DistributedBatchSampler, self).__init__(sampler, batch_size,
                                                      drop_last)
        if rank == -1:
            assert False, 'should not be here'
            rank = torch.distributed.get_rank()
        self.rank = rank
        self.world_size = world_size
        self.sampler.wrap_around = 0
        self.wrap_around = 0
        self.wrap_last = wrap_last
        self.start_iter = 0
        self.interleave = interleave

    def __iter__(self):
        batch = []
        i = 0
        for idx in self.data_iterator(self.sampler, wrap_around=False):
            batch.append(idx)
            if len(batch) == self.batch_size:
                tbatch = self._batch(batch)
                if i >= self.start_iter:
                    yield tbatch
                    self.start_iter = 0
                i += 1
                batch = []
        batch_len = len(batch)
        if batch_len > 0 and not self.drop_last:
            if self.wrap_last:
                self.sampler.wrap_around -= (self.batch_size)
                self.wrap_around += (len(batch))
                self.wrap_around %= self.batch_size
            yield self._batch(batch)
        if self.wrap_last:
            self.sampler.wrap_around += self.batch_size

    def data_iterator(self, _iter, wrap_around=False):
        """iterates through data and handles wrap around"""
        for i, idx in enumerate(_iter):
            if i < self.wrap_around % self.batch_size:
                continue
            if wrap_around:
                self.wrap_around += 1
                self.wrap_around %= self.batch_size
            yield idx

    def _batch(self, batch):
        """extracts samples only pertaining to this worker's batch"""
        if self.interleave:
            return batch[self.rank:self.batch_size:self.world_size]
        start = self.rank * self.batch_size // self.world_size
        end = (self.rank + 1) * self.batch_size // self.world_size
        return batch[start:end]



