import jieba
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging

jieba.setLogLevel(logging.INFO)  # 关闭打印jieba库

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 预测函数
def predict_next_words(model, sentence, vocab, idx_to_word, seq_length, device='cpu', num_words=1):
    model.eval()
    words = list(jieba.cut(sentence))
    word_indices = [vocab.get(word, vocab['<unk>']) for word in words]

    # 初始化输入序列
    input_seq = word_indices[-seq_length:]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    predicted_words = []

    for _ in range(num_words):
        with torch.no_grad():
            output = model(input_tensor)
            next_word_idx = output[0, -1].argmax(dim=-1).item()
            predicted_words.append(idx_to_word.get(next_word_idx, '<unk>'))

            # 更新输入序列，滑动窗口
            input_seq = input_seq[1:] + [next_word_idx]
            input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    return ' '.join(predicted_words)


# 一个简单的文本生成模型
class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out)
        return out


class TextDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.inputs[idx], dtype=torch.long)
        target_tensor = torch.tensor(self.targets[idx], dtype=torch.long)
        return input_tensor, target_tensor


def create_dataloader(inputs, targets, batch_size):
    dataset = TextDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
