import torch
from model import LanguageModel
from utils import encode, decode, vocab_size, device

model = LanguageModel(vocab_size)

model.load_state_dict(torch.load('model_weights.pth'))
model = model.to(device)
def respond_to_prompt(n_tokens):
    prompt = input()
    prompt = torch.tensor(encode(prompt))
    prompt = torch.stack((prompt, ), dim=0)
    out = model.generate(prompt, n_tokens)[0].tolist()
    print(decode(out))

while True:
    respond_to_prompt(100)