import os


class Config(object):
    """RNNLM模型配置项"""
    embedding_dim = 200  # 词向量维度

    rnn_type = 'LSTM'  # 支持RNN/LSTM/GRU
    hidden_dim = 200  # 隐藏层维度
    num_layers = 2  # RNN 层数

    dropout = 0.5  # 丢弃概率
    tie_weights = True  # 是否绑定参数

    batch_size = 10  # 每一批数据量
    seq_len = 30  # 序列长度

    clip = 0.25  # 用于梯度规范化
    learning_rate = 20  # 初始学习率

    num_epochs = 3  # 迭代轮次
    log_interval = 100  # 每隔多少个批次输出一次状态
    save_interval = 1  # 每个多少个轮次保存一次参数


class PathConfig(object):
    index_name_path = 'json_File/index_name.json'
    index_content_path = 'json_File/index_content.json'
    index_keyphrase_path = 'json_File/index_keyphrase.json'
    index_keywords_path = 'json_File/index_keywords.json'
    index_summarization_path = 'json_File/index_summarization.json'
    keywords_index_path = 'json_File/keywords_index.json'
    name_index_path = 'json_File/name_index.json'


class CorpusConfig(object):
    save_path = 'pickle_file/corpus.pkl'


STOPWORDS = os.path.join("data", "stop_words.txt")