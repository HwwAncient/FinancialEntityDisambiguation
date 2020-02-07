#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
File: source/preprocess.py
"""

import operator
import pickle
from functools import reduce
from pyhanlp import *
from source.inputters.voc import *
from source.inputters.dataset import *


def analysis_json(file_name):
    """

    @param file_name: json文件名
    @return:一个list存有json处理完后每个句子分成前半段 mention 后半段的格式
    """
    with open(file_name, encoding="utf-8") as load_f:
        load_dict = json.load(load_f)

    file_text = []
    for text_index in range(len(load_dict)):
        text = load_dict[text_index]['text']
        lab_result = load_dict[text_index]['lab_result']
        mention = []
        for mention_index in range(len(lab_result)):
            mention.append(lab_result[mention_index]['mention'])
        for replace_index in range(len(mention)):
            text = text.replace(mention[replace_index], " ")
        file_text.append(text.split())

    return reduce(operator.add, file_text)


def analysis_detail(file_name, voc, save_file_name):
    """

    @param file_name:要处理的json文件目录
    @param voc: 选择处理json的voc类
    @param save_file_name: 处理完后写的文件目录
    @return:用于将原本的train里面的数据向量化
    """
    with open(file_name, encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
        for text_index in range(len(load_dict)):
            text = load_dict[text_index]['text']
            lab_result = load_dict[text_index]['lab_result']
            mention = []
            for mention_index in range(len(lab_result)):
                mention.append(lab_result[mention_index]['mention'])
                load_dict[text_index]['mention'] = voc.word2index[lab_result[mention_index]['mention']]
                load_dict[text_index]['kb_id'] = test_kb(lab_result[mention_index]['kb_id'])
                load_dict[text_index]['offset'] = lab_result[mention_index]['offset']
                del load_dict[text_index]['lab_result']
            for replace_index in range(len(mention)):
                text = text.replace(mention[replace_index], " " + mention[replace_index] + " ")
            text = text.split()
            words = get_words(text)
            load_dict[text_index]['text'] = index_from_sentence(voc, words)
        with open(save_file_name, "w") as f:
            json.dump(load_dict, f)


def index_from_sentence(voc, words):
    """

    @param voc: voc类
    @param words:包含需要转为数字的list
    @return:向量化的数据
    """
    text_list = [SOS_token] + [voc.word2index[word] for word in words] + [EOS_token]
    return list(text_list + [PAD_token] * (80 - len(text_list)))


def test_kb(num):
    if num == -1:
        return 0
    else:
        return 1


def get_words(text_box):
    """

    @param text_box: 装有待分句子的list
    @return: 句子分成词之后的list
    """
    words = []
    for sentence in text_box:
        for term in HanLP.segment(sentence):
            words.append(term.word)
    return words


def get_mentions(file_name):
    """

    @param file_name: mentions存的位置
    @return: mentions的list
    """
    file_mention = []
    for line in open(file_name, encoding='utf-8'):
        file_mention.append(line[:-1])
    return file_mention


def save_class_data(file_name, class_data):
    """

    @param file_name: 写的目录名
    @param class_data: 要保存的类的实例
    @return: 用于将处理好的voc类保存
    """
    output_hal = open(file_name, 'wb')
    bi_str = pickle.dumps(class_data)
    output_hal.write(bi_str)
    output_hal.close()


def read_class_data(file_name):
    with open(file_name, 'rb') as file:
        return pickle.loads(file.read())


def get_max_length(file_name):
    with open(file_name, encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
        voc_length = []
        for text_index in range(len(load_dict)):
            voc_length.append(len(load_dict[text_index]['text']))
        return max(voc_length)


def text2voc(json_file_name, mention_file_name, save_file_name, voc_name):
    """

    @param json_file_name: 用于构建voc的text数据的json文件
    @param mention_file_name: 村粗mention的文件
    @param save_file_name: 保存的voc文件目录
    @param voc_name: 构建的voc类的名称
    @return: 用于将原json到保存好的voc类
    """
    json_file_name = json_file_name
    mention_file_name = mention_file_name
    voc_test = Voc(voc_name)
    text_box = analysis_json(json_file_name)
    word_box = get_words(text_box)
    mentions = get_mentions(mention_file_name)
    voc_test.add_word(mentions)
    voc_test.add_word(word_box)
    save_class_data(save_file_name, voc_test)


if __name__ == '__main__':
    # root_dir = "../data/resource/"
    # text2voc(root_dir + "train_and_dev.json", root_dir + "A14_company_name.txt", root_dir + "voc.pkl", "voc_test")
    # max_length = get_max_length("vectorized_data.json")
    # print(max_length)
    # voc = Voc("voc")
    # voc = read_class_data("voc.pkl")
    # analysis_detail("train.json", voc, "vectorized_data.json")
    text_data = Dataset("../../data/resource/train.json", "../../data/resource/vectorized_data.json")
    data_loader = text_data.create_batches()
    for i_batch, sample_batched in enumerate(data_loader):
        print(i_batch)
        print(sample_batched['text'])
        if i_batch == 2:
            break
