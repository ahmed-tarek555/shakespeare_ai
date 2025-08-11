import torch
from model import LanguageModel
from utils import decode

model = LanguageModel(65)

model.load_state_dict(torch.load('model_weights.pth'))

idx = torch.zeros(1,1, dtype=torch.long)

out = model.generate(idx, 1000)[0].tolist()

print(decode(out))