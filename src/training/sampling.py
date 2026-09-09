import torch
from torch.utils.data import Sampler


class FinalFullTrainSampler(Sampler):
    """Weighted draws initially; one shuffled permutation per final epoch."""

    def __init__(self, weighted_sampler, dataset_size, first_full_epoch):
        self.weighted_sampler = weighted_sampler
        self.dataset_size = dataset_size
        self.first_full_epoch = first_full_epoch
        self.epoch = 0
        self.generator = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.dataset_size if self.epoch >= self.first_full_epoch else len(self.weighted_sampler)

    def __iter__(self):
        if self.epoch >= self.first_full_epoch:
            return iter(torch.randperm(self.dataset_size, generator=self.generator).tolist())
        self.weighted_sampler.generator = self.generator
        return iter(self.weighted_sampler)
