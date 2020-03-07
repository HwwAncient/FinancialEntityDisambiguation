#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
File: source/inputters/dataset.py
"""

from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils
import torch


class LSTMDataset(Dataset):
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
        input_describe = data['describe']
        target = data['target']

        return input_text, input_mention, input_offset, input_describe, target

    @staticmethod
    def collate_fn(batch):
        """
        collate_fn
        """
        batch.sort(key=lambda x: x[2], reverse=True)

        input_text_left = []
        left_length = []
        input_text_right = []
        right_length = []
        input_mention = []
        describe_length = []
        input_describe = []
        target = []

        for elem in batch:
            offset = elem[2]
            left = elem[0][:offset+1]
            right = list(reversed(elem[0][offset:]))
            left_length.append(len(left))
            right_length.append(len(right))
            input_text_left.append(torch.tensor(left))
            input_text_right.append(torch.tensor(right))
            input_mention.append(elem[1])
            input_describe.append(torch.tensor(elem[3]))
            describe_length.append(len(elem[3]))
            target.append(float(elem[4]))

        out = {}

        out['input_left'] = rnn_utils.pad_sequence(input_text_left, batch_first=True)
        out['input_right'] = rnn_utils.pad_sequence(input_text_right, batch_first=True)
        out['input_mention'] = torch.tensor(input_mention)
        out['left_length'] = torch.tensor(left_length)
        out['right_length'] = torch.tensor(right_length)
        out['input_describe'] = rnn_utils.pad_sequence(input_describe, batch_first=True)
        out['describe_length'] = torch.tensor(describe_length)
        out['target'] = torch.tensor(target).unsqueeze(1).unsqueeze(2)

        return out

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
                            shuffle=shuffle, collate_fn=self.collate_fn, num_workers=2)
        return loader
