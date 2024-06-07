import torch
from RNN.model import SimpleRNNModel, predict_next_words
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_size = 100
hidden_size = 128
seq_length = 16

save_path = r'F:\AI_Learn\The-most-basic-language-prediction-model\parameter\model.pth'
vocab_path = r'F:\AI_Learn\The-most-basic-language-prediction-model\data\Vocabulary\fantastic\fantastic0.csv'
if __name__ == '__main__':

    vocab_df = pd.read_csv(vocab_path, encoding='gbk', header=None)
    vocab = {row[0]: int(row[1]) for _, row in vocab_df.iterrows()}
    idx_to_word = {index: word for word, index in vocab.items()}

    # 重新加载训练好的模型
    vocab_size = len(vocab)
    model = SimpleRNNModel(vocab_size, embed_size, hidden_size)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)

    # 交互式输入
    while True:
        input_sentence = input("请输入文本（输入#结束）：")
        if input_sentence == '#':
            break
        predicted_text = predict_next_words(model, input_sentence, vocab, idx_to_word, seq_length, device=device,
                                            num_words=5)

        print(f'输入: {input_sentence}')
        print(f'预测: {predicted_text}')
        print('本次生成结束...')