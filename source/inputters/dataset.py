#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
File: source/inputters/dataset.py
"""

from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils
import torch


class DefaultDataset(Dataset):
    """
    Dataset
    @param data ```List[Dict]```
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        input_text = data['text']
        input_mention = data['mention']
        input_offset = data['offset']
        target = data['target']

        return input_text, input_mention, input_offset, target

    @staticmethod
    def collate_fn(batch):
        """
        collate_fn
        """
        batch.sort(key=lambda x: len(x[0]), reverse=True)

        input_text = []
        input_mention = []
        input_offset = []
        target = []

        for elem in batch:
            input_text.append(torch.tensor(elem[0]))
            input_mention.append(elem[1])
            input_offset.append(elem[2])
            target.append(elem[3])

        input_text = rnn_utils.pad_sequence(input_text, batch_first=True)
        input_mention = torch.tensor(input_mention)
        input_offset = torch.tensor(input_offset)
        target = torch.tensor(target)

        return input_text, input_mention, input_offset, target

    def create_batches(self, batch_size=128, shuffle=True):
        """
        create_batches
                loader = DataLoader(dataset=self,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=self.collate_fn(device),
                            pin_memory=False)
        """
        loader = DataLoader(dataset=self,
                            batch_size=batch_size,
                            shuffle=shuffle, collate_fn=self.collate_fn)
        return loader
