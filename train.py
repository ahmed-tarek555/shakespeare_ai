import torch
from model import LanguageModel
from utils import vocab_size, encode, text, device

n_iter = 5000
lr = 3e-4

data = torch.tensor(encode(text))

n = int(len(data)*0.9)
train_data = data[:n]

model = LanguageModel(vocab_size)
model = model.to(device)
model._train(train_data, n_iter, lr)

torch.save(model.state_dict(), 'model_weights.pth')
