# -*- coding: utf-8 -*-


import jieba
import numpy as np
from nltk.cluster.util import cosine_distance
from config import STOPWORDS as STOPWORDS_PATH

MIN_SEQ_LEN = 0


def load_stopwords(file_path):
    with open(file_path, encoding='utf-8') as f:
        return [line.strip() for line in f]


def split_doc(doc, stopwords=None):
    if not stopwords:
        stopwords = []

    sentences = []
    cut_sentences = []
    origin_sentences = []

    while len(doc) > 0:
        for i in range(len(doc)):
            if doc[i] in ['。', '！', '?', '？', '……']:
                sentences.append(doc[:i+1])
                doc = doc[i+1:]
                break
            if i == len(doc) - 1:
                sentences.append(doc[:])
                doc = ''
                break
    for sent in sentences:
        if len(sent) > MIN_SEQ_LEN:
            cut_sentence = [word for word in jieba.lcut(sent) if word not in stopwords]
            if len(cut_sentence) > 0:
                cut_sentences.append(cut_sentence)
                origin_sentences.append(sent)
    return origin_sentences, cut_sentences


def sentence_similarity(sent1, sent2):
    """
    计算两个句子之间的相似性
    :param sent1:
    :param sent2:
    :return:
    """
    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in sent1:
        vector1[all_words.index(word)] += 1

    for word in sent2:
        vector2[all_words.index(word)] += 1
    # print(sent1)
    # print(sent2)
    # print('vector1:{}'.format(vector1))
    # print('vector2:{}'.format(vector2))
    # cosine_distance 越大越不相似
    return 1-cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences):
    """
    构建相似矩阵
    :param sentences:
    :return:
    """
    S = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            S[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2])
    # 将矩阵正则化
    for idx in range(len(S)):
        if S[idx].sum() == 0:
            continue
        S[idx] /= S[idx].sum()

    return S


def pagerank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs(new_P - P).sum()
        if delta <= eps:
            return new_P
        P = new_P

class TextRankSummarization:
    def __init__(self, ratio):
        self.ratio = ratio
        self.stopwords = load_stopwords(STOPWORDS_PATH)

    def analysis(self, doc):
        origin_sentences, cut_sentences = split_doc(doc, stopwords=self.stopwords)

        S = build_similarity_matrix(cut_sentences)

        sentences_ranks = pagerank(S)

        sentences_ranks = [item[0] for item in sorted(enumerate(sentences_ranks), key=lambda item: -item[1])]

        selected_sentences_index = sorted(sentences_ranks[:int(len(origin_sentences)*self.ratio)])

        summary = []
        for idx in selected_sentences_index:
            summary.append(origin_sentences[idx])

        return ''.join(summary)


# file_string = """足球可是我们学校的特色呢！我们九龙小学可是被称为“足球学校”的呦！在我们的学校到处都有足球的“身影”呢！瞧！那个足球形状的垃圾桶；墙也被涮上了一个大大的足球，可真是醒目！每个班级里更是有着大大小小的足球，那真是人手一个哦！当然，除了体育课，我们还有别的学校所没有的特殊课‘足球课’！虽然我的运动神经不够好，可是我和足球却有着好多好多的趣闻趣事哦！听我来说说吧！一次，我们在上足球课，这节课老师安排我们足球比赛，男生一组，女生一组，我当仁不让就是女生组的后卫了。嘿！嘿！嘿！这可是我最爱的位置，因为不用全场的跟着球跑哦，理由吗？我就不说了……通常这个时候我总是在球门边闲荡的。今天也是，我在球门边优哉游哉的走着，想着顺便看看我们和男生前场“厮杀”的怎样了，球有没有往我这里过来。你问我啊“为什么这么关心球有没有过来。”是吗？是因为啊！球如果过来了，肯定就会有一群人窝在一起抢球，我嘛……就在外围看着，然后，看球如果“跑”出来的话，我就赶紧去踢几脚，也算是恪守岗位了！这次照样球在我预料中跑了出来，可没想到我一个不注意，球打在了我的大腿上，我还来不及嚎叫，那球便被我腿的反弹力弹走了。本来快逼近我们球门的球被我弹到了中线，就在我还为我的腿按摩的时候，女生方传来了胜利的欢呼声！张晓蕾冲到我的面前兴奋的对着我说“胖胖，我们赢了，都是你的那个球，踢得真好！”“什么好啊？”我迷蒙的看着她，一头雾水的说。心想着，赢了是不是那5个俯卧撑就不用做了呢？下课后，同学们都夸我说踢得不错。正因为那足球“调皮”的一弹，我被同学们夸了好久。对！都是足球的功劳，我非常的开心，这算不算为球沾光呢？足球表现的团队配合力量，个人的小小突破也许就会给团队带来大转机。我感谢足球，同时也慢慢的喜欢起足球了！"""
# summ = TextRankSummarization(ratio=0.2)
# summary = summ.analysis(file_string)
# print(summary)