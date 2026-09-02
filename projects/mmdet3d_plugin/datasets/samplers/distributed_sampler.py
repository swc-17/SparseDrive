import math

import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from .sampler import SAMPLER

import pdb
import sys


class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def set_trace():
    ForkedPdb().set_trace(sys._getframe().f_back)


@SAMPLER.register_module()
class DistributedSampler(_DistributedSampler):
    def __init__(
        self, dataset=None, num_replicas=None, rank=None, shuffle=True, seed=0
    ):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle
        )
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        assert not self.shuffle
        if "data_infos" in dir(self.dataset):
            timestamps = [
                x["timestamp"] / 1e6 for x in self.dataset.data_infos
            ]
            vehicle_idx = [
                x["lidar_path"].split("/")[-1][:4]
                if "lidar_path" in x
                else None
                for x in self.dataset.data_infos
            ]
        else:
            timestamps = [
                x["timestamp"] / 1e6
                for x in self.dataset.datasets[0].data_infos
            ] * len(self.dataset.datasets)
            vehicle_idx = [
                x["lidar_path"].split("/")[-1][:4]
                if "lidar_path" in x
                else None
                for x in self.dataset.datasets[0].data_infos
            ] * len(self.dataset.datasets)

        sequence_splits = []
        for i in range(len(timestamps)):
            if i == 0 or (
                abs(timestamps[i] - timestamps[i - 1]) > 4
                or vehicle_idx[i] != vehicle_idx[i - 1]
            ):
                sequence_splits.append([i])
            else:
                sequence_splits[-1].append(i)

        indices = []
        perfix_sum = 0
        split_length = len(self.dataset) // self.num_replicas
        seq_i = 0
        while seq_i < len(sequence_splits):
            if perfix_sum >= (self.rank + 1) * split_length:
                break
            if perfix_sum >= self.rank * split_length:
                indices.extend(sequence_splits[seq_i])
            perfix_sum += len(sequence_splits[seq_i])
            seq_i += 1
        # Integer split drops len(dataset) % num_replicas frames. Last rank
        # takes the leftover sequences so eval asserts (e.g. navtest 12146)
        # still see every sample.
        if self.rank == self.num_replicas - 1:
            while seq_i < len(sequence_splits):
                indices.extend(sequence_splits[seq_i])
                seq_i += 1

        self.num_samples = len(indices)
        return iter(indices)
