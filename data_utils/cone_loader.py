#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

class ConeTrainSet(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, idx):
        return self.data[idx]
    
class ConeTestSet(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class ConeDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()
    def __len__(self):
        return len(self.batch_sampler.sampler)
    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
            
class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

