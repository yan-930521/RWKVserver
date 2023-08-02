import time
from torch import Tensor, allclose, save, load

import json
import asyncio
import argparse
import websockets

from rwkvlib.rwkv_cpp_model import RWKVModel
from rwkvlib.rwkv_cpp_shared_library import load_rwkv_shared_library
from rwkvlib.sampling import sample_logits
from rwkvlib.rwkv_tokenizer import get_tokenizer

from typing import List, Dict, Optional, Tuple
from flask import Flask, request, jsonify

# ======================================== Script settings ========================================
parser = argparse.ArgumentParser(description='rwkv ws ai vtuber server')
parser.add_argument('port', help='server port', type=int, default=3033)
args = parser.parse_args()

MODEL_PATH: str = "E:\LLMS\ChatRWKV\model\RWKV-4-World-7B_CN-Q5_1.bin"

# 20B, world
TOKENIZER: str = "world"

CPU_COUNT: int = 4
GPU_LAYER: int = 16

DEVICE: str = "cpu"

USER: str = "USER"
ASSISTANT: str = "樱氏"
INTERFACE: str = ":"

CACHE_NAME = "default_sakura_vtuber_v1_with_emotion"
CACHE_PATH = "./data/" + CACHE_NAME + ".pth"

INTERFACE: str = ":"

MODE: str = "CHAT"  # "QA"

# 傳來的歷史是否處理成字串
HISTORYMODE: str = "STRING"  # "STATE"

MAX_GENERATION_LENGTH: int = 250

# Sampling temperature. It could be a good idea to increase temperature when top_p is low.
TEMPERATURE: float = 1.5  # 0.8
# For better Q&A accuracy and less diversity, reduce top_p (to 0.5, 0.2, 0.1 etc.)
TOP_P: float = 0.6  # 0.5
# Penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
PRESENCE_PENALTY: float = 0.2
# Penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
FREQUENCY_PENALTY: float = 0.2

END_OF_LINE_TOKEN: int = 187
DOUBLE_END_OF_LINE_TOKEN: int = 535
END_OF_TEXT_TOKEN: int = 0

# =================================================================================================

class Cache:
    def __init__(self):
        self.cache: Tuple[Tensor, Tensor] = [None, None]
        self.tokens: List[int] = []
        self._setToken: bool = False
        self._set: bool = False
        self.touch()

    def setFromName(self, name: str = "default_cache", caches=None):
        if caches is None:
            return False
        cache = caches.get(name)
        if cache is not None and cache.hasSet():
            logits, state = cache.get()
            self.set(logits.clone(), state.clone())
            self.setToken(cache.tokens)
            self.touch()
            return True
        return False

    def setFromObject(self, cache):
        if cache is not None:
            logits, state = cache.cache
            self.set(logits.clone(), state.clone())
            self.setToken(cache.tokens)
            self.touch()
            return True
        return False

    def setToken(self, tokens: List[int]):
        self.touch()
        self._setToken = True
        self.tokens = tokens

    def pushSepuence(self, sepuence: str):
        self.touch()
        self.tokens += tokenizer_encode(sepuence)

    def pushTokens(self, tokens: List[int]):
        self.touch()
        self.tokens += tokens

    def getTokenIndex(self, new_tokens: List[int]):
        ln = len(new_tokens)  # [0, 1, 2, 3]-> 4
        lo = len(self.tokens)  # [0, 1, 2]   -> 3

        # 如果還沒讀過這串token，讓模型重新讀取一次
        if self.hasSetToken() is False or ln < lo:
            self.setToken(new_tokens)
            return 0

        for i in range(lo):
            if (new_tokens[i] != self.tokens[i]):
                return i
        return lo

    def set(self, out, state):
        self.touch()
        self._set = True
        self.cache = [out.clone(), state.clone()]

    def get(self):
        self.touch()
        if (self.hasSetToken()):
            out, state = self.cache
            return out.clone(), state.clone()
        return self.cache

    def hasSetToken(self):
        return self._setToken

    def hasSet(self):
        return self._set

    def touch(self):
        self.time = time.time()

# =================================================================================================

model: RWKVModel = None

tokenizer_decode = None
tokenizer_encode = None
nowState = None

vtuber: Cache = Cache()
default_cache = load(CACHE_PATH)

vtuber.setFromObject(default_cache)
# =================================================================================================


def load_model():
    global model, tokenizer_decode, tokenizer_encode
    library = load_rwkv_shared_library()
    print(f'System info: {library.rwkv_get_system_info_string()}')
    print('Loading RWKV model with gpu layer count', GPU_LAYER)

    model = RWKVModel(library, MODEL_PATH, CPU_COUNT, gpu_layers_count=GPU_LAYER)
    tokenizer_decode, tokenizer_encode = get_tokenizer(TOKENIZER)


def process_tokens(_tokens: List[int], new_line_logit_bias: float = 0.0) -> None:
    logits, state = vtuber.get()
    if len(_tokens) == 1:
        for _token in _tokens:
            logits, state = model.eval(_token, state, state, logits)

        logits[END_OF_LINE_TOKEN] += new_line_logit_bias

        vtuber.set(logits, state)
        return

    index: int = vtuber.getTokenIndex(_tokens)

    print("START AT: " + str(index) + " -> " + str(len(_tokens)), flush=True)
    for i in range(index, len(_tokens)):
        print("AT: " + str(i) + " -> " + str(len(_tokens)), flush=True, end="\r")
        logits, state = model.eval(_tokens[i], state, state, logits)
    print("END EVAL", flush=True)
    logits[END_OF_LINE_TOKEN] += new_line_logit_bias

    vtuber.set(logits, state)


def generateMessage(role, content):
    return (f"{role}{INTERFACE} " + content)


def remove_suffix(input_string, suffix):  # 兼容python3.8
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def getCompletionByTokens(tokens, top_p, temperature):
    token_count = MAX_GENERATION_LENGTH
    token_stop = [END_OF_TEXT_TOKEN]
    resultChat = ""
    resultTokens: List[int] = []

    print("TOKEN LEN: " + str(len(tokens)), flush=True)

    process_tokens(
        tokens,
        new_line_logit_bias=-999999999
    )

    print(generateMessage(ASSISTANT, ""), end="", flush=True)

    accumulated_tokens: List[int] = []
    token_counts: Dict[int, int] = {}

    for i in range(int(token_count)):
        logits, state = vtuber.get()

        for n in token_counts:
            logits[n] -= PRESENCE_PENALTY + token_counts[n] * FREQUENCY_PENALTY

        token: int = sample_logits(logits, temperature, top_p)

        if token in token_stop:
            break

        if token not in token_counts:
            token_counts[token] = 1
        else:
            token_counts[token] += 1

        vtuber.set(logits, state)
        vtuber.pushTokens([token])

        process_tokens([token])

        accumulated_tokens += [token]
        resultTokens += [token]

        decoded: str = tokenizer_decode(accumulated_tokens)

        if '\uFFFD' not in decoded:
            print(decoded[-1], end='', flush=True)
            resultChat = resultChat + decoded
            if resultChat.endswith('\n\n') or resultChat.endswith(f"{USER}{INTERFACE}") or resultChat.endswith(f"{ASSISTANT}{INTERFACE}"):
                resultChat = remove_suffix(
                    remove_suffix(
                        remove_suffix(
                            remove_suffix(resultChat, f"{USER}{INTERFACE}"), f"{ASSISTANT}{INTERFACE}"),
                        '\n'),
                    '\n')
                print()
                return resultChat, vtuber.tokens
        accumulated_tokens = []
    print()
    return resultChat, vtuber.tokens

# =================================================================================================

async def handler(websocket, path):
    try:
        while True:
            recv_str_data = await websocket.recv()
            # print(recv_str_data, flush=True)
            data = json.loads(recv_str_data)

            content = data.get("content")
            temperature = data.get("temperature")
            top_p = data.get("top_p")

            print("get content: " + content, flush=True)

            if content is not None:
                tokens = tokenizer_encode(content)

                reply, tokens = getCompletionByTokens(
                    tokens=tokens,
                    temperature=temperature,
                    top_p=top_p
                )

                respond_data = {
                    "respond": {
                        "content": reply
                    }
                }

                await websocket.send(json.dumps(respond_data))
    except websockets.exceptions.ConnectionClosedOK:
        print("Connection closed by the client.")
    except Exception as error:
        print("Connection closed cause by error.", error)
print("Starting AI VTUBER Process Service RWKV At Port:", args.port, flush=True)

load_model()

start_server = websockets.serve(handler, 'localhost', args.port)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
