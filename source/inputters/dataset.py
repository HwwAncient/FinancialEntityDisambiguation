#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
File: source/inputters/dataset.py
"""

from torch.utils.data import DataLoader, Dataset
import json
from source.utils.misc import Pack
from source.utils.misc import list2tensor


class Dataset(Dataset):
    """
    Dataset
    """
    def __init__(self, root_dir, vec_dir):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.vec_dir = vec_dir  # 处理后的文件
        with open(vec_dir, encoding="utf-8") as load_f:
            load_dict = json.load(load_f)
            self.text_detail = load_dict

    def __len__(self):
        return len(self.text_detail)

    def __getitem__(self, idx):
        return self.text_detail[idx]

    @staticmethod
    def collate_fn(device=-1):
        """
        collate_fn
        """
        def collate(data_list):
            """
            collate
            """
            batch = Pack()
            for key in data_list[0].keys():
                batch[key] = list2tensor([x[key] for x in data_list])
            if device >= 0:
                batch = batch.cuda(device=device)
            return batch
        return collate

    def create_batches(self, batch_size=128, shuffle=True, device=-1):
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
                            shuffle=shuffle)
        return loader
