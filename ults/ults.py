import codecs
import re

import jieba


def clean_text(text):
    # 去除字符串首尾的空白字符（包括空格、换行符等）
    text = text.strip()
    # 使用正则表达式去除标点和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 去除数字
    text = re.sub(r'\d+', '', text)
    # 去除字符串中间的多余空格
    text = ' '.join(text.split())
    # 使用jieba分词，并去除非中文字符
    words = jieba.cut(text)
    clean_words = [word for word in words if '\u4e00' <= word[0] <= '\u9fff']
    # 重新组合清洗后的词语
    cleaned_text = ''.join(clean_words)
    return cleaned_text


# 清洗文本并过滤空列表
def clean_and_filter(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        cleaned_text = clean_text(sentence)
        if cleaned_text:  # 确保清洗后的文本不为空
            cleaned_sentences.append(cleaned_text)
    return cleaned_sentences


# 映射输入到索引
def map_words_to_vocab(tokenized_sentences, vocab, unk_index):
    """
    将分词后的句子列表映射到词汇表中的索引。

    参数:
    tokenized_sentences (list of list of str): 分词后的句子列表。
    vocab (dict): 词汇表，键为单词，值为对应的索引。
    unk_index (int, optional): 未知单词的索引，如果提供，则用该索引替代未在词汇表中的单词。

    返回:
    list of list of int: 句子列表，其中每个单词被映射为词汇表中的索引。
    """
    mapped_sentences = []
    for words in tokenized_sentences:
        if unk_index is True:
            mapped_indices = [vocab.get(word, vocab['<unk>']) for word in words]
        else:
            mapped_indices = [vocab[word] for word in words if word in vocab]
        mapped_sentences.append(mapped_indices)
    return mapped_sentences


# 创建输入和标签
def create_input_target_pairs(tokenized_sentences, seq_length):
    """
    创建输入和标签对。

    参数:
    tokenized_sentences : 映射到词汇表中的句子列表。
    seq_length : 每个输入序列的长度。

    返回:
    tuple of lists: inputs 和 targets。
    """
    inputs = []
    targets = []
    for sentence in tokenized_sentences:
        for i in range(len(sentence) - seq_length):
            input_seq = sentence[i:i + seq_length]
            target_seq = sentence[i + 1:i + seq_length + 1]
            inputs.append(input_seq)
            targets.append(target_seq)
    return inputs, targets


# 检测文本编码格式
def detect_encoding(file_path):
    encodings = ['utf-8', 'gbk', 'gb2312', 'big5', 'latin1']
    for encoding in encodings:
        try:
            with codecs.open(file_path, 'r', encoding) as f:
                return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("No suitable encoding found for the file.")
