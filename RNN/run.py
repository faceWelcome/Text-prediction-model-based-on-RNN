import jieba
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from model import SimpleRNNModel, TextDataset
from ults.ults import map_words_to_vocab, create_input_target_pairs, clean_and_filter
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型、损失函数和优化器 parameters
embed_size = 100
batch_size = 150
hidden_size = 128
lr = 0.001
seq_length = 16

save_path = r'F:\AI_Learn\RNN_model\parameter\model.pth'  # 保存模型地址
file_path = r'F:\AI_Learn\RNN_model\data\语料\TestS.txt'  # 这里请选择你自己的文本数据 .txt .csv皆可
vocab_path = r'F:\AI_Learn\RNN_model\data\Vocabulary\fantastic\fantastic0.csv'  # 词汇表的位置

# 训练模块
if __name__ == '__main__':

    # 从CSV文件中读取词汇表
    # 你可以自己新建一个词汇表，可以直接运行ults.vocabularuMake.py来生成
    print('加载词汇表...')
    vocab_df = pd.read_csv(vocab_path, encoding='gbk', header=None)
    vocab = {row[0]: int(row[1]) for _, row in vocab_df.iterrows()}
    next_index = max(vocab.values()) + 1
    print('词汇表加载完成...')
    print(f'词汇表大小为:{len(vocab)}')

    # sentences = ['我有一个梦想', '我的梦想是飞到天上去']
    print('加载数据集...')
    with open(file_path, 'r', encoding='gbk') as f:  # encoding='utf-8'如果报错就修改utf-8
        sentences = f.readlines()
    sentences = clean_and_filter(sentences)
    print('加载完成...')

    print('进行分词...')
    # 分词
    tokenized_sentences = []
    for sentence in sentences:
        words = list(jieba.cut(sentence))
        tokenized_sentences.append(words)
    print('分词完成...')

    tokenized_sentences_index = map_words_to_vocab(tokenized_sentences, vocab, unk_index=True)

    src, trg = create_input_target_pairs(tokenized_sentences_index, seq_length)

    # 创建模型、损失函数和优化器
    vocab_size = len(vocab)
    model = SimpleRNNModel(vocab_size, embed_size, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    num_epochs = 100
    model.to(device)

    print('加载 dataloader...')
    dataset = TextDataset(src, trg)
    train_size = int(0.8 * len(dataset))  # 划分数据集以及验证集
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print('dataloader 加载完成')

    train_losses = []
    val_losses = []

    print('开始训练...')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for input_batch, target_batch in progress_bar:
            input_batch = input_batch.to(device).long()
            target_batch = target_batch.to(device).long()

            optimizer.zero_grad()
            output = model(input_batch)

            # 修改大小来执行loss计算
            output = output.view(-1, vocab_size)
            target_batch = target_batch.view(-1)

            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        print(f'本次epoch:{epoch+1}训练完成, 开始进行本轮验证...')
        # 计算验证损失
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_input_batch, val_target_batch in val_dataloader:
                val_input_batch = val_input_batch.to(device).long()
                val_target_batch = val_target_batch.to(device).long()

                val_output = model(val_input_batch)
                val_output = val_output.view(-1, vocab_size)
                val_target_batch = val_target_batch.view(-1)

                val_loss += criterion(val_output, val_target_batch).item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, 训练损失 Loss: {avg_train_loss}, 验证损失 Loss: {avg_val_loss}')

        # 计算困惑度
        train_ppl = np.exp(avg_train_loss)
        val_ppl = np.exp(avg_val_loss)
        print(f'本次训练困惑度为: {train_ppl}, 本次验证困惑度为: {val_ppl}')

    print('训练完成...')
    torch.save(model.state_dict(), save_path)
    print(f'成功保存模型到{save_path}')
