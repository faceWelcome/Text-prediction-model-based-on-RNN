import pandas as pd
import json

# 读取CSV文件, 可以是一个空的
df = pd.read_csv(r'F:\AI_Learn\RNN_model\data\Vocabulary\fantastic\fantastic0.csv', encoding='gbk')

# 假设CSV文件中有列名为word的列，包含词语数据
words = df['word'].dropna().unique()

# 创建词汇表，这里使用字典来存储，词语作为键，索引作为值
vocab = {word: i for i, word in enumerate(words)}

# 打印词汇表的大小
print(f"词汇表大小: {len(vocab)}")

# 保存词汇表到文件
with open(r'/data/vocabulary.csv', 'w', encoding='utf-8') as f:
    for word, index in vocab.items():
        f.write(f"{word} {index}\n")

# 将词汇表保存为JSON格式
# with open('vocab.json', 'w', encoding='utf-8') as f:
#     json.dump(vocab, f)
