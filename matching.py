import json
from config import PathConfig

index_content_path = PathConfig.index_content_path
index_keyphrase_path = PathConfig.index_keyphrase_path
index_name_path = PathConfig.index_name_path
index_summarization_path = PathConfig.index_summarization_path
index_keywords_path = PathConfig.index_keywords_path
name_index_path = PathConfig.name_index_path
keywords_index_path = PathConfig.keywords_index_path


class MatchTool:
    def __init__(self, index_content_path, index_keyphrase_path, index_name_path,
                 index_summarization_path, index_keywords_path, name_index_path, keywords_index_path):
        self.index_content = self.open_file(index_content_path)
        self.index_keyphrase = self.open_file(index_keyphrase_path)
        self.index_keywords = self.open_file(index_keywords_path)
        self.index_name = self.open_file(index_name_path)
        self.index_summarization = self.open_file(index_summarization_path)
        self.name_index = self.open_file(name_index_path)
        self.keywords_index = self.open_file(keywords_index_path)

    def open_file(self, path):
        print('Reading file {}......'.format(path))
        try:
            with open(path, 'r', encoding='utf-8') as f:
                json_file = json.load(f)
        except Exception as e:
            with open(path, 'r', encoding='gbk') as f:
                json_file = json.load(f)
        print('Finished Reading file {}......'.format(path))
        return json_file

    def index2content(self, index):
        return self.index_content[index]

    def index2keyphrase(self, index):
        return self.index_keyphrase[index]

    def index2keywords(self, index):
        return self.index_keywords[index]

    def index2name(self, index):
        return self.index_name[index]

    def index2summarization(self, index):
        return self.index_summarization[index]

    def name2index(self, index):
        return self.name_index[index] if self.name_index[index] is not None else False

    def keywords2index(self, index):
        return self.keywords_index[index] if self.keywords_index[index] is not None else False

    def search_by_keyword(self, word):
        index = self.keywords2index(word)
        if index is False:
            return None
        essay_dict_list = []
        for i in index:
            essay_dict = {}
            essay_dict['标题'] = self.index2name(i)
            essay_dict['文章内容'] = self.index2content(i)
            essay_dict['文章关键字'] = self.index2keywords(i)
            essay_dict['文章关键短语'] = self.index2keyphrase(i)
            essay_dict['文章摘要'] = self.index2summarization(i)
            essay_dict_list.append(essay_dict)
        return essay_dict_list

    def search_by_title(self, name):
        i = self.name2index(name)
        if i is False:
            return None
        essay_dict = {}
        essay_dict['标题'] = self.index2name(i)
        essay_dict['文章内容'] = self.index2content(i)
        essay_dict['文章关键字'] = self.index2keywords(i)
        essay_dict['文章关键短语'] = self.index2keyphrase(i)
        essay_dict['文章摘要'] = self.index2summarization(i)
        return essay_dict


match_tool = MatchTool(index_content_path, index_keyphrase_path, index_name_path,
                 index_summarization_path, index_keywords_path, name_index_path, keywords_index_path)
my_dict = match_tool.search_by_keyword('秋天')
print(my_dict)