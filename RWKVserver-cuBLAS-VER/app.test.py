import time
from torch import Tensor, allclose, save, load

from rwkvlib.rwkv_cpp_model import RWKVModel
from rwkvlib.rwkv_cpp_shared_library import load_rwkv_shared_library
from rwkvlib.sampling import sample_logits
from rwkvlib.rwkv_tokenizer import get_tokenizer

from typing import List, Dict, Optional, Tuple
from flask import Flask, request, jsonify

# ======================================== Script settings ========================================

MODEL_PATH: str = "E:\LLMS\ChatRWKV\model\RWKV-4-World-7B_CN-Q5_1.bin"

# 20B, world
TOKENIZER: str = "world"

CPU_COUNT: int = 4
GPU_LAYER: int = 16

DEVICE: str = "cpu"

USER: str = "User"
ASSISTANT: str = "Assistant"
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

    def setFromName(self, name: str = "default_cache", caches = None):
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


model: RWKVModel = None

tokenizer_decode = None
tokenizer_encode = None
nowState = None

cache: Cache = Cache()
caches: Dict[str, Cache] = {}
caches["default_cache"] = Cache()
# =================================================================================================


def load_model():
    global model, tokenizer_decode, tokenizer_encode
    library = load_rwkv_shared_library()
    print(f'System info: {library.rwkv_get_system_info_string()}')
    print('Loading RWKV model')

    model = RWKVModel(library, MODEL_PATH, CPU_COUNT, gpu_layer_count=GPU_LAYER)
    tokenizer_decode, tokenizer_encode = get_tokenizer(TOKENIZER)


def process_tokens(_tokens: List[int], new_line_logit_bias: float = 0.0) -> None:
    logits, state = cache.get()
    # logits2, state2 = None, None
    # print(logits, state)

    if len(_tokens) == 1:
        for _token in _tokens:
            logits, state = model.eval(_token, state, state, logits)

        logits[END_OF_LINE_TOKEN] += new_line_logit_bias

        cache.set(logits, state)
        return

    index: int = cache.getTokenIndex(_tokens)
    # print(_tokens)
    print("START AT: " + str(index) + " -> " + str(len(_tokens)), flush=True)
    for i in range(index, len(_tokens)):
        logits, state = model.eval(_tokens[i], state, state, logits)
    # for _token in _tokens:
    #    logits2, state2 = model.eval(_token, state2, state2, logits2)

    logits[END_OF_LINE_TOKEN] += new_line_logit_bias
    # logits2[END_OF_LINE_TOKEN] += new_line_logit_bias

    # print(logits2, state2)
    cache.set(logits, state)

    # logits2, state2 = cache.get()

    # print("logits == logits2", allclose(logits, logits2))
    # print("state == state2", allclose(state, state2))

def process_tokens_with_id(id: str, _tokens: List[int], new_line_logit_bias: float = 0.0) -> None:
    cache = caches.get(id)
    logits, state = cache.get()
    if len(_tokens) == 1:
        for _token in _tokens:
            logits, state = model.eval(_token, state, state, logits)

        logits[END_OF_LINE_TOKEN] += new_line_logit_bias

        cache.set(logits, state)
        return

    index: int = cache.getTokenIndex(_tokens)
    #　print(tokenizer_decode(_tokens))
    #　print(tokenizer_decode(cache.tokens))

    print("START AT: " + str(index) + " -> " + str(len(_tokens)), flush=True)
    for i in range(index, len(_tokens)):
        print("AT: " + str(i) + " -> " + str(len(_tokens)), flush=True, end="\r")
        logits, state = model.eval(_tokens[i], state, state, logits)
    print("END EVAL", flush=True)
    logits[END_OF_LINE_TOKEN] += new_line_logit_bias

    cache.set(logits, state)

def generateMessage(role, content):
    return (f"{role}{INTERFACE} " + content)


def remove_suffix(input_string, suffix):  # 兼容python3.8
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


def process_history(history, mode: str):
    if (mode is None):
        mode = HISTORYMODE
    if mode == "STRING" and len(history) > 0:
        tmp = []
        for i, old_chat in enumerate(history):
            if old_chat['role'] is not None:
                tmp.append(generateMessage(
                    old_chat['role'], old_chat['content']))

        history = '\n\n'.join(tmp) + "\n\n"
        return history
    return ""


def getCompletion(userInput: str, initPrompt: str, chat, history, history_mode, max_length, top_p, temperature, data):
    if (userInput is None):
        userInput = ""
    if (initPrompt is None):
        initPrompt = ""
    if (max_length is None):
        max_length = MAX_GENERATION_LENGTH
    if (history_mode is None):
        history_mode = HISTORYMODE
    if (temperature is None):
        temperature = TEMPERATURE
    if (top_p is None):
        top_p = TOP_P

    token_count = max_length
    token_stop = [END_OF_TEXT_TOKEN]
    resultChat = ""

    ctx = initPrompt

    if (history is not None and history_mode == "STRING"):
        ctx += process_history(history, history_mode)

    if (chat is True):
        ctx += generateMessage(USER, userInput.replace("\n\n", "\n")) + \
            "\n\n" + generateMessage(ASSISTANT, "")
    else:
        print("[RWKV raw mode] ", end="")
        ctx += userInput.replace("\n\n", "\n")

    print("PROMPT LEN: " + str(len(ctx)), flush=True)

    # ctx = ctx.strip()
    # print(f'{ctx}', end='')

    process_tokens(
        tokenizer_encode(ctx),
        new_line_logit_bias=-999999999
    )

    print(generateMessage(ASSISTANT, ""), end="", flush=True)

    accumulated_tokens: List[int] = []
    token_counts: Dict[int, int] = {}

    for i in range(int(token_count)):
        logits, state = cache.get()

        for n in token_counts:
            logits[n] -= PRESENCE_PENALTY + token_counts[n] * FREQUENCY_PENALTY

        # logits_backup = logits.clone()

        token: int = sample_logits(logits, temperature, top_p)

        # print("logits == logits_backup", allclose(logits, logits_backup))

        if token in token_stop:
            break

        if token not in token_counts:
            token_counts[token] = 1
        else:
            token_counts[token] += 1

        cache.set(logits, state)

        process_tokens([token])

        # Avoid UTF-8 display issues
        accumulated_tokens += [token]

        decoded: str = tokenizer_decode(accumulated_tokens)

        # print(decoded, flush=True)

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
                return resultChat
        accumulated_tokens = []
    print()
    return resultChat

def getCompletionByTokens(id, name, tokens, user, assistant, max_length, top_p, temperature, data):
    if (user is None):
        user = USER
    if (assistant is None):
        assistant = ASSISTANT
    
    if(id not in caches):
        c = Cache()
        c.setFromName(name, caches)
        caches[id] = c
    print("CACHE ID: ", id, flush=True)
    print("TMP NAME: ", name, flush=True)
    cache = caches.get(id)

    # print(tokenizer_decode(cache.tokens), flush=True)

    token_count = max_length
    token_stop = [END_OF_TEXT_TOKEN]
    resultChat = ""
    resultTokens: List[int] = []

    print("TOKEN LEN: " + str(len(tokens)), flush=True)

    process_tokens_with_id(
        id,
        tokens,
        new_line_logit_bias=-999999999
    )

    print(generateMessage(assistant, ""), end="", flush=True)

    accumulated_tokens: List[int] = []
    token_counts: Dict[int, int] = {}

    for i in range(int(token_count)):
        logits, state = cache.get()

        for n in token_counts:
            logits[n] -= PRESENCE_PENALTY + token_counts[n] * FREQUENCY_PENALTY

        token: int = sample_logits(logits, temperature, top_p)

        if token in token_stop:
            break

        if token not in token_counts:
            token_counts[token] = 1
        else:
            token_counts[token] += 1

        cache.set(logits, state)
        cache.pushTokens([token])

        process_tokens_with_id(id, [token])
        
        accumulated_tokens += [token]
        resultTokens += [token]

        decoded: str = tokenizer_decode(accumulated_tokens)

        if '\uFFFD' not in decoded:
            print(decoded[-1], end='', flush=True)
            resultChat = resultChat + decoded
            if resultChat.endswith('\n\n') or resultChat.endswith(f"{user}{INTERFACE}") or resultChat.endswith(f"{assistant}{INTERFACE}"):
                resultChat = remove_suffix(
                    remove_suffix(
                        remove_suffix(
                            remove_suffix(resultChat, f"{user}{INTERFACE}"), f"{assistant}{INTERFACE}"),
                        '\n'),
                    '\n')
                print()
                return resultChat, cache.tokens
        accumulated_tokens = []
    print()
    return resultChat, cache.tokens

# =================================================================================================

# history = []

# load_model()

'''
while True:
    user_input = input('> ' + generateMessage(USER, ""))
    msg = user_input.replace('\\n', '\n').strip()

    if (msg == "+i"):
        print(history)
        continue

    reply = getCompletion(
        userInput=msg,
        initPrompt=initPrompt,
        chat=True,
        max_length=200,
        history=history,
        history_mode="STRING",
        temperature=None,
        top_p=None,
        data=None
    )

    history += [{
        "role": USER,
        "content": msg
    }, {
        "role": ASSISTANT,
        "content": reply
    }]
'''

app = Flask(__name__)


@app.route("/")
def hello():
    return "Sakura Assiatant"

@app.route("/gettokens", methods=['POST', 'GET'])
def gettokens():
    if request.method == 'POST':
        data = request.get_json()
        content = data["content"]
        tokens = []
        if(content is not None):
            tokens = tokenizer_encode(content)

        respond_data = {
            "respond": {
                "tokens": tokens
            }
        }
        return jsonify(respond_data)
    else:
        return "/gettokens"

@app.route("/setdefaulttoken", methods=['POST', 'GET'])
def setdefaulttoken():
    if request.method == 'POST':
        data = request.get_json()
        tokens = data["tokens"]
        success = False
        if(tokens is not None):
            process_tokens_with_id(
                "default_cache",
                tokens,
            )
            success = True

        respond_data = {
            "respond": {
                "success": success
            }
        }
        return jsonify(respond_data)
    else:
        return "/setdefaulttoken"

@app.route("/savetokens", methods=['POST', 'GET'])
def savetokens():
    if request.method == 'POST':
        data = request.get_json()
        tokens = data["tokens"]
        name = data["name"]
        if(name not in caches):
            c = Cache()
            caches[name] = c
        print("ADD CACHE NAME: ", name)
        process_tokens_with_id(name, tokens)

        save(caches[name], "./data/" + name + ".pth")

        respond_data = {
            "respond": {
                "name": name,
                "tokens": tokens
            }
        }
        return jsonify(respond_data)
    else:
        return "/savetokens"

@app.route("/savecache", methods=['POST', 'GET'])
def savecache():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get("name")
        file = data.get("file")
        success = False
        if(name in caches and file is not None):
            print("Save CACHE NAME: ", name)
            save(caches[name], "./data/" + file + ".pth")
            success = True

        respond_data = {
            "respond": {
                "success": success
            }
        }
        return jsonify(respond_data)
    else:
        return "/savecache"
    
@app.route("/loadtokens", methods=['POST', 'GET'])
def loadtokens():
    if request.method == 'POST':
        data = request.get_json()
        name = data["name"]

        c = load("./data/" + name + ".pth")
        caches[name] = Cache()
        success = caches[name].setFromObject(c)

        if(success):
            respond_data = {
                "respond": {
                    "success": True,
                    "tokens": caches[name].tokens
                }
            }
        else:
            respond_data = {
                "respond": {
                    "success": False,
                    "tokens": []
                }
            }
        return jsonify(respond_data)
    else:
        return "/loadtokens"

@app.route("/completionbytokens", methods=['POST', 'GET'])
def completionByTokens():
    if request.method == 'POST':
        data = request.get_json()
        name=data["name"]
        
        reply, tokens = getCompletionByTokens(
            id=data["id"],
            name=data["name"],
            tokens=data["tokens"],
            user=data["user"],
            assistant=data["assistant"],
            max_length=data["max_length"],
            temperature=data["temperature"],
            top_p=data["top_p"],
            data=data["data"]
        )

        # print(tokenizer_decode(tokens))

        respond_data = {
            "respond": {
                "content": reply,
                "tokens": tokens
            }
        }
        return jsonify(respond_data)
    else:
        return "/completionbytokens"

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=3033)