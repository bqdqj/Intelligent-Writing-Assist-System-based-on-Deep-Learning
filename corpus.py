#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import torch
import json
import tqdm
import pickle
from config import CorpusConfig


class Dictionary(object):
    """
    词汇表，将文本中的词转换为数字id表示。
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    """
    文本预处理，获取词汇表，并将字符串文本转换为数字序列。
    """

    def __init__(self, path=None, save_dict=False, read_from_dict=False):
        self.dictionary = Dictionary()
        if read_from_dict:
            self.read_dict_from_pickle()
        else:
            self.train = self.tokenize(path)
        if save_dict:
            self.save_dict_to_pickle()

    def tokenize(self, path):
        """文本符号化，转换为数字id表示。"""
        assert os.path.exists(path)

        # 将新词加入到词汇表中
        print('Adding words to vocab......')
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            load_json_file = json.load(f)
            for doc in tqdm.tqdm(load_json_file.values()):
                words = list(doc.strip()) + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # 将字符转换为数字
        print('Transforming words to index......')
        with open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            load_json_file = json.load(f)
            for doc in tqdm.tqdm(load_json_file.values()):
                words = list(doc.strip()) + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def save_dict_to_pickle(self):
        print("Saving corpus to {} ......".format(CorpusConfig.save_path))
        with open(CorpusConfig.save_path, 'wb') as f:
            s = pickle.dumps(self.dictionary)
            f.write(s)
            f.close()
        print('Saving File Successfully!')

    def read_dict_from_pickle(self):
        print("Reading corpus from {} ......".format(CorpusConfig.save_path))
        with open(CorpusConfig.save_path, 'rb') as f:
            self.dictionary = pickle.loads(f.read())
        print("Reading File Successfully!")

    def __repr__(self):
        return "Corpus length: %d, Vocabulary size: %d" % (self.train.size(0), len(self.dictionary))