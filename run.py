import torch
from model import LanguageModel
from utils import encode, decode, vocab_size, device, block_size

model = LanguageModel(vocab_size)

model.load_state_dict(torch.load('model_weights_pre.pth', map_location=torch.device(device)))
model = model.to(device)

init = torch.ones(1, block_size, dtype=torch.long)
out = model.generate(init, 1000).squeeze(0).tolist()
print(decode(out[block_size:]))