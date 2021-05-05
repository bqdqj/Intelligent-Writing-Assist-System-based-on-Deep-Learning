import os
import re
from key_words_extraction import TfidfKeywords
from key_phrases_extraction import KeyPhraseExtraction
from text_summarization import TextRankSummarization
import json
import tqdm

essay_list = []
for i in os.listdir(r'F:\Datasets\zwwn_data'):
    essay_list.append(i)

# 需要用到的字典
name_index_dict = {}
index_name_dict = {}
index_content_dict = {}
keywords_index_dict = {}
index_keywords_dict = {}
index_keyphrase_dict = {}
index_summarization_dict = {}


summ = TextRankSummarization(ratio=0.3)
tfidf_keywords = TfidfKeywords(delete_stopwords=True, topK=20, withWeight=False)
key_phrase_extractor = KeyPhraseExtraction(topk=100)
for essay_name in tqdm.tqdm(essay_list):
    file = open(os.path.join(r'F:\Datasets\zwwn_data',essay_name), 'rb')
    lines = file.readlines()
    file_index = essay_name.split('.')[0]
    file_string = ''

    # 循环获取文章内容，将文章标题设成字典
    for i, line in enumerate(lines):
        if i == 1:
            line = line.decode()
            line = re.split(pattern=r'[\s]', string=line.strip())
            essay_topic = line[0].split('：')[-1]
            name_index_dict[essay_topic] = file_index
            index_name_dict[file_index] = essay_topic
        elif i < 7:
            pass
        else:
            line = line.decode()
            line = re.split(pattern=r'[\s]', string=line.strip())
            file_string += line[0]
    index_content_dict[file_index] = file_string

    # 关键字词提取
    keywords = tfidf_keywords.keywords(file_string)
    for key_word in keywords:
        if key_word in keywords_index_dict.keys():
            keywords_index_dict[key_word].append(file_index)
        else:
            keywords_index_dict[key_word] = []
            keywords_index_dict[key_word].append(file_index)
    index_keywords_dict[file_index] = keywords

    # 关键短语提取
    key_phrase = key_phrase_extractor.key_phrase_extraction(file_string)
    key_phrase_list = []
    for key, value in key_phrase.items():
        if value > 0.1:
            key_phrase_list.append(key)
    index_keyphrase_dict[file_index] = key_phrase_list

    # 文本摘要
    # print(file_string)
    summary = summ.analysis(file_string.replace('”', ''))
    index_summarization_dict[file_index] = summary
    # print(summary)


name_index_json_str = json.dumps(name_index_dict, ensure_ascii=False, indent=4)
with open('name_index.json', 'w', encoding='utf-8') as json_file:
    json_file.write(name_index_json_str)

index_name_json_str = json.dumps(index_name_dict, ensure_ascii=False, indent=4)
with open('index_name.json', 'w', encoding='utf-8') as json_file:
    json_file.write(index_name_json_str)

index_content_json_str = json.dumps(index_content_dict, ensure_ascii=False, indent=4)
with open('index_content.json', 'w', encoding='utf-8') as json_file:
    json_file.write(index_content_json_str)

keywords_index_json_str = json.dumps(keywords_index_dict, ensure_ascii=False, indent=4)
with open('keywords_index.json', 'w', encoding='utf-8') as json_file:
    json_file.write(keywords_index_json_str)

index_keywords_json_str = json.dumps(index_keywords_dict, ensure_ascii=False, indent=4)
with open('index_keywords.json', 'w', encoding='utf-8') as json_file:
    json_file.write(index_keywords_json_str)

index_keyphrase_json_str = json.dumps(index_keyphrase_dict, ensure_ascii=False, indent=4)
with open('index_keyphrase.json', 'w') as json_file:
    json_file.write(index_keyphrase_json_str)

index_summarization_json_str = json.dumps(index_summarization_dict, ensure_ascii=False, indent=4)
with open('index_summarization.json', 'w', encoding='utf-8') as json_file:
    json_file.write(index_summarization_json_str)