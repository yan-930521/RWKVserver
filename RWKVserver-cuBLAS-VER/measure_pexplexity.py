# Measures perplexity and per-token latency of an RWKV model on a given text file.
# Perplexity is defined here as exp() of average cross-entropy loss.
# Usage: python measure_pexplexity.py C:\rwkv.cpp-169M.bin C:\text.txt 1024

import os
import time
import torch
import argparse
from rwkvlib.rwkv_cpp_model import RWKVModel
from rwkvlib.rwkv_cpp_shared_library import load_rwkv_shared_library
from rwkvlib.rwkv_tokenizer import get_tokenizer

parser = argparse.ArgumentParser(description='rwkv ws ai vtuber server')
parser.add_argument('cpu', help='cpu count', type=int, nargs="?", default=1)
parser.add_argument('gpu', help='gpu count', type=int, nargs="?", default=1)
args = parser.parse_args()

MODEL_PATH: str = "E:\LLMS\ChatRWKV\model\RWKV-4-World-7B_CN-Q5_1.bin"

# 20B, world
TOKENIZER: str = "world"

CPU_COUNT: int = 1
DEVICE: str = "cpu"


print('Loading text')
text = """扮演扮演扮演扮演扮演扮演扮演扮演"""

_, tokenizer_encode = get_tokenizer(TOKENIZER)

tokens = tokenizer_encode(text)

token_count: int = len(tokens)
print(f'{token_count} tokens in the text')

# ---

def format_loss(loss: torch.Tensor) -> str:
    return str(['%.3f' % (loss[i].item(),) for i in range(len(loss))]).replace('\'', '')[1:-1]

def format_loss_with_perplexity(loss: torch.Tensor) -> str:
    return f'loss [{format_loss(loss)}], perplexity {"%.3f" % (torch.exp(loss[0]).item(),)}'

# ---

model: RWKVModel = RWKVModel(
    load_rwkv_shared_library(),
    MODEL_PATH,
    args.cpu,
    gpu_layers_count=args.gpu
)

logits, state = None, None

loss_sum: torch.Tensor = torch.tensor([0.0])
loss_count: int = 0

start: float = time.time()

run_count: int = token_count - 1

for i in range(run_count):
    token: int = tokens[i]
    target: int = tokens[i + 1]

    logits, state = model.eval(token, state, state, logits)

    
    losses = torch.tensor([
        torch.nn.functional.cross_entropy(logits, torch.tensor(target, dtype=torch.long), reduction='none').item()
    ])

    loss_sum += losses
    loss_count += 1

    if run_count <= 5 or i % (run_count // 10) == 0:
        avg_loss_so_far = loss_sum / loss_count

        duration: float = time.time() - start
        duration_per_token: float = duration / (i + 1)
        runs_remaining: int = run_count - i - 1
        duration_remaining: int = int(runs_remaining * duration_per_token)

        print(f'Token #{i}/{token_count}, '
              f'{int(100.0 * i / token_count)}%, '
              f'ETA {duration_remaining // 60} m {duration_remaining % 60} s', end='')

        if loss_count > 0:
            print(f', averages so far: {format_loss_with_perplexity(avg_loss_so_far)}')
        else:
            print()

print()
print(f'averages: {format_loss_with_perplexity(loss_sum / loss_count)}, '
      f'latency {int((time.time() - start) * 1000 / run_count)} ms per token')
