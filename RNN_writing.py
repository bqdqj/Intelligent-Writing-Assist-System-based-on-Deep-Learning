import os
import torch

from RNN_LM import RNNModel
from corpus import Corpus
from config import Config

use_cuda = torch.cuda.is_available()
model_dir = 'checkpoints/index_content'


def get_last_model(model_dir):
    model_list = os.listdir(model_dir)
    last_model = model_list[0]
    for model in model_list:
        index = model.split('.')[0].split('_')[-1]
        if int(index) > int(last_model.split('.')[0].split('_')[-1]):
            last_model = model
    return last_model


def generate_writing(input_word_list, word_len=50, save_dict=False, read_from_dict=True):
    corpus = Corpus(save_dict=save_dict, read_from_dict=read_from_dict)
    config = Config()
    config.vocab_size = len(corpus.dictionary)

    model = RNNModel(config)

    model_file = get_last_model(model_dir)
    model_path = os.path.join(model_dir, model_file)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print('Successfully loaded model {}'.format(model_path))

    if use_cuda:
        model = model.cuda()

    word_list = generate(model, input_word_list, corpus, word_len=word_len)
    return word_list


def tranform_words2idx(input_word_str, corpus):
    input_word_list = str_to_list(input_word_str)
    index_list = []
    for word in input_word_list:
        index = corpus.dictionary.word2idx[word]
        index_list.append(index)
    return index_list


def str_to_list(input_word_str):
    return [char for char in input_word_str]


def generate(model, input_word_list, corpus, word_len=200, temperature=1.0):
    """生成一定数量的文本，temperature结合多项式分布可增添抽样的多样性。"""
    model.eval()
    hidden = model.init_hidden(1)  # batch_size为1
    with torch.no_grad():
        inputs = torch.tensor(tranform_words2idx(input_word_list, corpus), dtype=torch.long)  # 随机选取一个字作为开始
        inputs = torch.unsqueeze(inputs, 1)
        if use_cuda:
            inputs = inputs.cuda()

        word_list = [word for word in input_word_list]
        for i in range(word_len):  # 逐字生成
            output, hidden = model(inputs, hidden)
            word_weights = output.squeeze().data.div(temperature).exp().cpu()

            # 基于词的权重，对其再进行一次抽样，增添其多样性，如果不使用此法，会导致常用字的无限循环
            word_idx = torch.multinomial(word_weights, 1)[0]
            inputs = torch.tensor([word_idx]).unsqueeze(0)
            if use_cuda:
                inputs = inputs.cuda()
            word = corpus.dictionary.idx2word[word_idx]
            word_list.append(word)
    return word_list


word_list = generate_writing('我最喜欢')
print(''.join(word_list))