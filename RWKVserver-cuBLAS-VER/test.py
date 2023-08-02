import torch
from rwkvlib.rwkv_tokenizer import get_tokenizer
from app import Cache

tokenizer_decode, tokenizer_encode = get_tokenizer("world")
c = torch.load("./data/default_sakura_vtuber_v1_with_emotion.pth")
print(tokenizer_decode(c.tokens))