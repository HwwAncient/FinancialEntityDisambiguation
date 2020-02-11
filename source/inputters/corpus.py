#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/inputters/corpus.py
"""


import torch.nn as nn

from source.inputters.voc import *
from source.inputters.dataset import *

WORD_VEC_DEPTH = 300


class Corpus(object):
    """
    Corpus
    语料库基类
    实现语料库基本的方法，包括加载，存储，构建语料库，以及构建batch等
    @param data_prefix 存储数据前缀
    @param min_freq 语料库中单个词汇最小出现数
    @param data_dir 处理完成的数据存储文件夹
    @param max_vocab_size 词汇库最大数量
    """
    def __init__(self,
                 data_prefix='demo',
                 min_freq=0,
                 data_dir="\\data",
                 max_vocab_size=None):
        self.data_dir = data_dir
        self.data_prefix = data_prefix
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        # 数据存储路径
        prepared_data_file = data_prefix + ".data.pt"
        prepared_vocab_file = data_prefix + ".vocab.pt"
        prepared_embeds_file = data_prefix + ".embeds.pt"
        self.prepared_data_file = os.path.join(data_dir, prepared_data_file)
        self.prepared_vocab_file = os.path.join(data_dir, prepared_vocab_file)
        self.prepared_embeds_file = os.path.join(data_dir, prepared_embeds_file)

        # 词典
        self.vocab = Voc('vocab')

        # 词嵌入
        self.embeds = None

        # 其他 todo 还没理解
        self.filter_pred = None
        self.sort_fn = None
        self.data = None

    def load(self):
        """
        从文件直接读取已经处理好的数据
        """
        if not (os.path.exists(self.prepared_data_file) and
                os.path.exists(self.prepared_vocab_file)):
            self.build()
        self.vocab.load(self.prepared_vocab_file)
        self.load_data(self.prepared_data_file)
        self.load_embeds()

    def reload(self, data_type='test'):
        """
        重新处理并读取数据
        @param 要读取数据类别
        """
        data_raw = self.read_data(data_type="test")
        data_examples = self.build_data(data_raw)
        self.data[data_type] = Dataset(data_examples)

        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_data(self, prepared_data_file=None):
        """
        从处理好文件加载数据
        @param prepared_data_file 数据来源
        """
        prepared_data_file = prepared_data_file or self.prepared_data_file
        print("Loading prepared data from {} ...".format(prepared_data_file))

        data = torch.load(prepared_data_file)
        self.data = {'train': Dataset(data['train']), 'test': Dataset(data['test'])}
        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def read_data(self, data_type='train', data_file=None):
        """
        在原文件中读取数据，每一条数据读取为一个字典
        @return ``List[Dict]``
                Dict{
                    'text': 句子,
                    'mention': 匹配实体,
                    'offset': 实体偏移,
                    'target' :结果
                    }
        """
        data_list = []
        file_path = data_file or os.path.join(self.data_dir, self.data_prefix + '.' + data_type)

        # log
        print("Reading data from '{}' ...".format(file_path))

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                datas = line.split('\t')
                data = {
                    'text': datas[0],
                    'mention': datas[1],
                    'offset': int(datas[2]),
                    'target': int(datas[3])
                }
                data_list.append(data)
        return data_list

    def build_data(self, data):
        """

        @param data_type:
        @return:
        """
        _data = []
        for dict in data:
            _dict = {}
            words = tokenizer(dict['text'])
            text = []
            target_offset = dict['offset']
            for word in words:
                index = self.vocab.word2index[word]
                text.append(index)
            _dict['text'] = text
            _dict['mention'] = self.vocab.word2index[dict['mention']]
            _dict['offset'] = target_offset
            _dict['target'] = dict['target']
            _data.append(_dict)
        return _data

    def build_vocab(self, data):
        """
        构建词汇表
        @param data ``List[Dict]``
        """
        if self.vocab is None:
            self.vocab = Voc('vocab')

        for dict in data:
            self.vocab.add_sentence(dict['text'])
            self.vocab.add_word([dict['mention']])

    def load_embeds(self):
        """
        加载词嵌入
        @return:
        """
        embeds_data = []
        print("Loading word embeds from '{}'".format(self.prepared_embeds_file))
        with open(self.prepared_embeds_file, 'r', encoding='utf-8') as f:
            for line in f:
                attr = line.split()
                attr_ = [float(i) for i in attr]
                embeds_data.append(attr_[1:])

        self.embeds = nn.Embedding(len(embeds_data), WORD_VEC_DEPTH)
        embeds = torch.tensor(embeds_data)
        self.embeds.weight.data.copy_(embeds)

    def build(self):
        """
        build
        """
        print("Start to build corpus!")

        print("Reading data ...")
        train_raw = self.read_data(data_type="train")
        test_raw = self.read_data(data_type="test")

        print("Building vocabulary ...")
        self.build_vocab(train_raw)
        self.build_vocab(test_raw)

        print("Building TRAIN data ...")
        train_data = self.build_data(train_raw)
        print("Building TEST data ...")
        test_data = self.build_data(test_raw)

        print("Loading word embedding ...")
        self.load_embeds()

        self.data = {"train": Dataset(train_data),
                     "test": Dataset(test_data)}

        print("Saving prepared vocab ...")
        self.vocab.save(self.prepared_vocab_file)
        print("Saved prepared vocab to '{}'".format(self.prepared_vocab_file))
        print("Saving prepared data ...")
        torch.save(self.data, self.prepared_data_file)
        print("Saved prepared data to '{}'".format(self.prepared_data_file))

    def create_batches(self, batch_size, data_type="train",
                       shuffle=False, device=None):
        """
        create_batches
        返回对应数据的 DataLoader
        """
        try:
            data = self.data[data_type]
            data_loader = data.create_batches(batch_size, shuffle)
            return data_loader
        except KeyError:
            raise KeyError("Unsported data type: {}!".format(data_type))

    def transform(self, batch_size,
                  data_type="test", shuffle=False):
        """
        Transform raw text from data_file to Dataset and create data loader.
        通过原始数据，直接生成 DataLoader
        """
        raw_data = self.read_data(data_type=data_type)
        examples = self.build_data(raw_data)
        data = Dataset(examples)
        data_loader = data.create_batches(batch_size, shuffle)
        return data_loader





# class SrcTgtCorpus(Corpus):
#     """
#     SrcTgtCorpus
#     """
#     def __init__(self,
#                  data_dir,
#                  data_prefix,
#                  min_freq=0,
#                  max_vocab_size=None,
#                  min_len=0,
#                  max_len=100,
#                  embed_file=None,
#                  share_vocab=False):
#         super(SrcTgtCorpus, self).__init__(data_dir=data_dir,
#                                            data_prefix=data_prefix,
#                                            min_freq=min_freq,
#                                            max_vocab_size=max_vocab_size)
#         self.min_len = min_len
#         self.max_len = max_len
#         self.share_vocab = share_vocab
#
#         self.SRC = TextField(tokenize_fn=tokenize,
#                              embed_file=embed_file)
#         if self.share_vocab:
#             self.TGT = self.SRC
#         else:
#             self.TGT = TextField(tokenize_fn=tokenize,
#                                  embed_file=embed_file)
#
#         self.fields = {'src': self.SRC, 'tgt': self.TGT}
#
#         def src_filter_pred(src):
#             """
#             src_filter_pred
#             """
#             return min_len <= len(self.SRC.tokenize_fn(src)) <= max_len
#
#         def tgt_filter_pred(tgt):
#             """
#             tgt_filter_pred
#             """
#             return min_len <= len(self.TGT.tokenize_fn(tgt)) <= max_len
#
#         self.filter_pred = lambda ex: src_filter_pred(ex['src']) and tgt_filter_pred(ex['tgt'])
#
#     def read_data(self, data_file, data_type="train"):
#         """
#         read_data
#         """
#         data = []
#         filtered = 0
#         with open(data_file, "r", encoding="utf-8") as f:
#             for line in f:
#                 src, tgt = line.strip().split('\t')[:2]
#                 data.append({'src': src, 'tgt': tgt})
#
#         filtered_num = len(data)
#         if self.filter_pred is not None:
#             data = [ex for ex in data if self.filter_pred(ex)]
#         filtered_num -= len(data)
#         print(
#             "Read {} {} examples ({} filtered)".format(len(data), data_type.upper(), filtered_num))
#         return data
#
#
# class KnowledgeCorpus(Corpus):
#     """
#     KnowledgeCorpus
#     """
#     def __init__(self,
#                  data_dir,
#                  data_prefix,
#                  min_freq=0,
#                  max_vocab_size=None,
#                  min_len=0,
#                  max_len=100,
#                  embed_file=None,
#                  share_vocab=False,
#                  with_label=False):
#         super(KnowledgeCorpus, self).__init__(data_dir=data_dir,
#                                               data_prefix=data_prefix,
#                                               min_freq=min_freq,
#                                               max_vocab_size=max_vocab_size)
#         self.min_len = min_len
#         self.max_len = max_len
#         self.share_vocab = share_vocab
#         self.with_label = with_label
#
#         self.SRC = TextField(tokenize_fn=tokenize,
#                              embed_file=embed_file)
#         if self.share_vocab:
#             self.TGT = self.SRC
#             self.CUE = self.SRC
#         else:
#             self.TGT = TextField(tokenize_fn=tokenize,
#                                  embed_file=embed_file)
#             self.CUE = TextField(tokenize_fn=tokenize,
#                                  embed_file=embed_file)
#
#         if self.with_label:
#             self.INDEX = NumberField()
#             self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE, 'index': self.INDEX}
#         else:
#             self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE}
#
#         def src_filter_pred(src):
#             """
#             src_filter_pred
#             """
#             return min_len <= len(self.SRC.tokenize_fn(src)) <= max_len
#
#         def tgt_filter_pred(tgt):
#             """
#             tgt_filter_pred
#             """
#             return min_len <= len(self.TGT.tokenize_fn(tgt)) <= max_len
#
#         self.filter_pred = lambda ex: src_filter_pred(ex['src']) and tgt_filter_pred(ex['tgt'])
#
#     def read_data(self, data_file, data_type="train"):
#         """
#         read_data
#         """
#         data = []
#         with open(data_file, "r", encoding="utf-8") as f:
#             for line in f:
#                 if self.with_label:
#                     src, tgt, knowledge, label = line.strip().split('\t')[:4]
#                     filter_knowledge = []
#                     for sent in knowledge.split(''):
#                         filter_knowledge.append(' '.join(sent.split()[:self.max_len]))
#                     data.append({'src': src, 'tgt': tgt, 'cue': filter_knowledge, 'index': label})
#                 else:
#                     src, tgt, knowledge = line.strip().split('\t')[:3]
#                     filter_knowledge = []
#                     for sent in knowledge.split(''):
#                         filter_knowledge.append(' '.join(sent.split()[:self.max_len]))
#                     data.append({'src': src, 'tgt': tgt, 'cue':filter_knowledge})
#
#         filtered_num = len(data)
#         if self.filter_pred is not None:
#             data = [ex for ex in data if self.filter_pred(ex)]
#         filtered_num -= len(data)
#         print(
#             "Read {} {} examples ({} filtered)".format(len(data), data_type.upper(), filtered_num))
#         return data
