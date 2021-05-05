# coding=gbk
import jieba
import jieba.analyse
import re
import numpy as np


class KeyPhraseExtraction:
    def __init__(self, topk=50, method='tfidf', with_word=True):
        """
        :param topk: ����ǰ���ٹؼ������ɶ���
        :param method: tfidf / textrank
        :param with_word: ����������Ƿ�����ؼ���
        """
        self.topk = topk
        self.method = method
        self.with_word = with_word

    def cut_sentences(self, text):
        """�ı��־䣬Ȼ��ִ�"""
        sentences = re.findall(".*?[������]", text)
        cut_sentences = [jieba.lcut(sent) for sent in sentences]
        return cut_sentences

    def key_words_extraction(self, text):
        """��ȡ�ؼ���"""
        keywords_score = []
        if self.method == 'tfidf':
            keywords_score = jieba.analyse.extract_tags(text, topK=self.topk, withWeight=True)
        elif self.method == 'textrank':
            keywords_score = jieba.analyse.textrank(text, topK=self.topk, withWeight=True)
        return {word: score for word, score in keywords_score}

    def key_phrase_extraction(self, text):
        keyword_score = self.key_words_extraction(text)
        keywords = keyword_score.keys()
        cut_sentences = self.cut_sentences(text)
        # print(keywords)
        # �����ڵĹؼ��ʽ���ƴ��
        key_phrase = []
        for sent in cut_sentences:
            temp = []
            for word in sent:
                if word in keywords:
                    temp.append(word)
                else:
                    if len(temp) > 1:
                        if temp not in key_phrase:
                            key_phrase.append(temp)
                    temp = []

        # ����֮����ܴ���������Ϣ�����й���
        key_phrase_filter = []
        for phrase in key_phrase:
            flag = False
            for item in key_phrase_filter:
                if len(set(phrase) & set(item)) >= min(len(set(phrase)), len(set(item)))/2.0:
                    flag = True
                    break
            if not flag:
                    key_phrase_filter.append(phrase)

        # �����︳ֵȨ��, ���ö��������������ؼ���
        keyphrase_weight = {''.join(phrase[-3:]): np.mean([keyword_score[word] for word in phrase[-3:]])
                            for phrase in key_phrase_filter}

        if self.with_word:
            key_phrase_str = '|'.join(keyphrase_weight)
            for word, weight in keyword_score.items():
                if word not in key_phrase_str:
                    keyphrase_weight[word] = weight
        keyphrase_weight = dict(sorted(keyphrase_weight.items(), key=lambda x: x[1], reverse=True)[:self.topk])

        return keyphrase_weight



