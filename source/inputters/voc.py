#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from hanlp import *

"""
File: source/inputters/voc.py
"""

# Default word tokens
PAD_token = 0  # Used for padding short sentences
# SOS_token = 1  # Start-of-sentence token
# EOS_token = 2  # End-of-sentence token
# UNK_token = 3  # unknown token
#
# tokenizer = None
#
# def getTokenizer(tokenizer):

tokenizer = hanlp.load("PKU_NAME_MERGED_SIX_MONTHS_CONVSEG")


class Voc:
    """
    Voc 词典类
    @param name 词典名称
    """
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD"}
        self.num_words = 4  # Count SOS, EOS, PAD, UNK


    def add_sentence(self, sentence):
        self.add_word(tokenizer(sentence))

    def add_word(self, word_list):
        for word in word_list:
            if word not in self.word2index:
                self.word2index[word] = self.num_words
                self.word2count[word] = 1
                self.index2word[self.num_words] = word
                self.num_words += 1
            else:
                self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD"}
        self.num_words = 4  # Count default tokens

        for word in keep_words:
            self.add_word(word)

    def load(self, file_path):
        """
        从文件加载词典
        @param file_path: 文件路径
        @return:
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                attr = line.split()
                index = int(attr[0])
                word = attr[1]
                self.word2index[word] = index
                self.word2count[word] = 1
                self.index2word[index] = word
