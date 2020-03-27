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
import torch

from source.inputters.voc import *
from source.inputters.dataset import DefaultDataset
from source.utils.log import *

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
                 logger,
                 dataset=DefaultDataset,
                 data_prefix='demo',
                 source_prefix='source',
                 min_freq=0,
                 data_dir="./data",
                 max_vocab_size=None):
        self.data_dir = data_dir
        self.data_prefix = data_prefix
        self.source_prefix = source_prefix
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.dataset = dataset
        self.logger = logger

        # 数据存储路径
        prepared_data_file = data_prefix + ".data.pt"
        prepared_embeds_file = data_prefix + ".embeds.pt"
        prepared_mention_file = data_prefix + ".mention.pt"
        self.prepared_data_file = os.path.join(data_dir, prepared_data_file)
        self.prepared_embeds_file = os.path.join(data_dir, prepared_embeds_file)
        self.prepared_mention_file = os.path.join(data_dir, prepared_mention_file)

        # 词典
        self.vocab = Voc('vocab')

        # 实体
        self.mention2id = {}
        self.id2describe = {}

        # 词嵌入
        self.embeds = None

        # 其他 todo 还没理解
        self.filter_pred = None
        self.sort_fn = None
        self.data = None

    def load(self, prepared_data_file=None):
        """
        从文件直接读取已经处理好的数据
        """
        prepared_data_file = prepared_data_file or self.prepared_data_file
        if not (os.path.exists(prepared_data_file) and
                os.path.exists(self.prepared_embeds_file)):
            self.logger.info("Source file does not exist, start to build...")
            self.build()

        self.logger.info("Source file build finished, start to load...")
        self.logger.info("Loading vocab data from '{}'".format(self.prepared_mention_file))
        self.vocab.load(self.prepared_embeds_file)
        self.load_data(self.prepared_data_file)
        self.load_embeds()

    def reload(self, data_type='test'):
        """
        重新处理并读取数据
        @param 要读取数据类别
        """
        data_raw = self.read_data(data_type="test")
        data_examples = self.build_data(data_raw)
        self.data[data_type] = self.dataset(data_examples)

        self.logger.info("Number of examples: {}-{}".format(k.upper(), len(v)) for k, v in self.data.items())

    def load_data(self, prepared_data_file=None):
        """
        从处理好文件加载数据
        @param prepared_data_file 数据来源
        """
        prepared_data_file = prepared_data_file or self.prepared_data_file
        self.logger.info("Loading prepared data from {} ...".format(prepared_data_file))

        data = torch.load(prepared_data_file)
        self.data = {'train': self.dataset(data['train']),
                     'train.large': self.dataset(data['train.large']),
                     'test.large': self.dataset(data['test.large']),
                     'test': self.dataset(data['test'])}
        self.logger.info("Number of examples:" + "".join([" {}-{} ".format(k.upper(), len(v)) for k, v in self.data.items()]))

    def read_mention(self):

        path = os.path.join(self.data_dir, 'source.mention')

        mention_dict = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                attrs = line.split('\t')
                id = attrs[0]
                name = attrs[1]
                full_name = attrs[2]
                index = attrs[3]
                describe = attrs[4]

                company = {}
                company['id'] = id
                company['full_name'] = full_name
                company['index'] = index
                company['describe'] = describe

                if name not in mention_dict:
                    mention_dict[name] = [company]
                else:
                    mention_dict[name].append(company)

        return mention_dict

    def build_mention(self, mention_dict):
        for mention in mention_dict:
            companys = mention_dict[mention]
            indexs = []
            for company in companys:
                describe = company['describe']
                text = tokenizer(describe)
                out = []
                for word in text:
                    if word in self.vocab.word2index:
                        index = self.vocab.word2index[word]
                        out.append(index)

                indexs.append(company['id'])
                self.id2describe[company['id']] = out

            self.mention2id[mention] = indexs

    def load_mention(self, prepared_mention_file=None):
        prepared_data_file = prepared_mention_file or self.prepared_mention_file
        self.logger.info("Loading prepared data from {} ...".format(prepared_data_file))

        data = torch.load(prepared_data_file)
        self.mention2id = data['mention2id']
        self.id2describe = data['id2describe']

    def read_data(self, data_type='train', data_file=None):
        """
        在原文件中读取数据，每一条数据读取为一个字典
        @return ``List[Dict]``
                Dict{
                    'text': 句子,
                    'mention': 匹配实体,
                    'offset': 实体偏移,
                    'target' :结果,
                    }
        """
        data_list = []
        file_path = data_file or os.path.join(self.data_dir, self.source_prefix + '.' + data_type)

        # log
        self.logger.info("Reading data from '{}' ...".format(file_path))

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                datas = line.split('\t')
                data = {
                    'text': datas[0],
                    'mention': datas[1],
                    'offset': int(datas[2]),
                    'target': int(datas[3]),
                    'company_id': int(datas[4])
                }
                data_list.append(data)
        return data_list

    def build_data(self, data):
        """
        @param data: ``List[Dict]``

        @return:data: Dict{
                    'text': 句子,
                    'mention': 匹配实体,
                    'offset': 实体偏移,
                    'target' : 结果,
                    'describe': 实体描述
                    }
        """
        _data = []

        for dict in data:
            offset = int(dict['offset'])
            mention_len = len(dict['mention'])

            prefix = dict['text'][:offset]
            suffix = dict['text'][offset + mention_len:]

            prefix_list = tokenizer(prefix)
            suffix_list = tokenizer(suffix)
            mention = dict['mention']

            prefix_index = []
            suffix_index = []

            mention_index = [self.vocab.word2index[mention]]

            for word in prefix_list:
                if word in self.vocab.word2index:
                    index = self.vocab.word2index[word]
                    prefix_index.append(index)

            for word in suffix_list:
                if word in self.vocab.word2index:
                    index = self.vocab.word2index[word]
                    suffix_index.append(index)
            company_list = self.mention2id[dict['mention']]

            for company_id in company_list:
                _dict = {}
                _dict['text'] = prefix_index + mention_index + suffix_index
                _dict['mention'] = self.vocab.word2index[dict['mention']]
                _dict['offset'] = len(prefix_index)
                _dict['describe'] = self.id2describe[company_id]
                _dict['target'] = dict['target'] if str(company_id) == str(dict['company_id']) else 0
                _data.append(_dict)

        return _data

    def load_embeds(self):
        """
        加载词嵌入
        @return:
        """
        embeds_data = [[0.0 for _ in range(WORD_VEC_DEPTH)]]
        self.logger.info("Loading word embeds from '{}'".format(self.prepared_embeds_file))
        with open(self.prepared_embeds_file, 'r', encoding='utf-8') as f:
            for line in f:
                attr = line.split()
                attr_ = [float(i) for i in attr[2:]]
                embeds_data.append(attr_)
        self.embeds = nn.Embedding(len(embeds_data), WORD_VEC_DEPTH)
        embeds = torch.tensor(embeds_data)
        self.embeds.weight.data.copy_(embeds)

    def build(self):
        """
        build
        """
        self.logger.info("Start to build corpus!")

        self.logger.info("Loading vocabulary ...")
        self.vocab.load(self.prepared_embeds_file)

        self.logger.info("Reading mention data ...")
        mention = self.read_mention()
        self.logger.info("Building mention data ...")
        self.build_mention(mention)

        self.logger.info("Reading TRAIN data ...")
        train_raw = self.read_data(data_type="train")
        train_large_raw = self.read_data(data_type="train.more.large")


        self.logger.info("Reading TEST data ...")
        test_large_raw = self.read_data(data_type="test.more.large")
        test_raw = self.read_data(data_type="test")

        self.logger.info("Building TRAIN data ...")
        train_data = self.build_data(train_raw)
        train_large_data = self.build_data(train_large_raw)

        self.logger.info("Building TEST data ...")
        test_large_data = self.build_data(test_large_raw )
        test_data = self.build_data(test_raw)

        self.logger.info("Loading word embedding ...")
        self.load_embeds()

        data = {"train": train_data,
                "train.large": train_large_data,
                "test.large": test_large_data,
                "test": test_data}

        self.data = {"train": self.dataset(train_data),
                     "train.large": self.dataset(train_large_data),
                     "test.large": self.dataset(test_large_data),
                     "test": self.dataset(test_data)}

        mention = {"mention2id": self.mention2id,
                   "id2describe": self.id2describe}

        self.logger.info("Saving prepared data ...")
        torch.save(data, self.prepared_data_file)
        self.logger.info("Saved prepared data to '{}'".format(self.prepared_data_file))
        self.logger.info("Saving prepared mention ...")
        torch.save(mention, self.prepared_mention_file)
        self.logger.info("Saved prepared mention to '{}'".format(self.prepared_mention_file))


    def create_batches(self, batch_size, data_type="train",
                       shuffle=True):
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
        data = self.dataset(examples)
        data_loader = data.create_batches(batch_size, shuffle)
        return data_loader

    def data_size(self, data_type="train"):
        return len(self.data[data_type])


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
