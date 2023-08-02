import os
import time
import hashlib
import struct
import torch
from typing import Dict, List, Tuple
from rwkvlib.rwkv_cpp_model import RWKVModel

DEFAULT_PATH = './RWKV_state_cache.bin'

SAVE_EVERY_N_TOKENS = 64

class RWKV_Cache:

    def __init__(self, file_path: str = DEFAULT_PATH):
        self.file_path = file_path
        self.cache_persistent: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.cache_transient: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.dirty = False

        if os.path.isfile(file_path):
            start = time.time()
            self.cache_persistent = torch.load(file_path, map_location='cpu')
            self.cache_transient = self.cache_persistent.copy()
            print('Loading cache took %.3f sec, %d entries' % (time.time() - start, len(self.cache_persistent)))

    def is_cached(self, model: RWKVModel, context_tokens: List[int], token: int) -> bool:
        cache_key = RWKV_Cache._cache_key(context_tokens, model, token)
        return self.cache_transient.get(cache_key) is not None

    # Returns copied tensors, they are safe to modify.
    def get(self, model: RWKVModel, context_tokens: List[int], token: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = RWKV_Cache._cache_key(context_tokens, model, token)
        cached_state = self.cache_transient.get(cache_key)
        out, new_state = cached_state
        return out.clone(), new_state.clone()

    def forward(self, model: RWKVModel, context_tokens: List[int], token: int, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = RWKV_Cache._cache_key(context_tokens, model, token)

        cached_state = self.cache_transient.get(cache_key)

        if cached_state is not None:
            out, new_state = cached_state

            return out.clone(), new_state.clone()
        else:
            out, new_state = model.forward(token, state)

            cache_value = (out.clone(), new_state.clone())

            self.cache_transient[cache_key] = cache_value

            if len(context_tokens) % SAVE_EVERY_N_TOKENS == 0:
                self.cache_persistent[cache_key] = cache_value
                self.dirty = True

            return out, new_state

    # Returns copied tensors, they are safe to modify.
    def preprocess_prompt(self, model: RWKVModel, tokens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        token_count = len(tokens)
        print(f'{token_count} tokens in prompt')

        if token_count == 0:
            raise ValueError('Empty prompt is not supported')

        out, state = None, None

        # Find longest prefix for which we have saved state
        longest_prefix: List[int] = []

        for i in range(token_count):
            token_index = token_count - i - 1
            longest_prefix = tokens[:token_index]
            token = tokens[token_index]

            if self.is_cached(model, longest_prefix, token):
                out, state = self.get(model, longest_prefix, token)
                break

        remaining = tokens[len(longest_prefix):]
        remaining_count = len(remaining)

        if remaining_count > 0:
            print(f'Processing {remaining_count} remaining prompt tokens')

            start = time.time()

            cache_key = longest_prefix

            for i in range(remaining_count):
                out, state = self.forward(model, cache_key, remaining[i], state)

                cache_key += [remaining[i]]

                if remaining_count < 5 or i % (remaining_count // 5) == 0:
                    print(f'{i}/{remaining_count}')

            delay = time.time() - start

            print('Took %.3f sec, %d ms per token' % (delay, delay / remaining_count * 1000))

        return out.clone(), state.clone()

    def save_if_dirty(self) -> None:
        if not self.dirty:
            return

        start = time.time()
        torch.save(self.cache_persistent, self.file_path + '.tmp')
        os.replace(self.file_path + '.tmp', self.file_path)
        print('\nSaving cache took %.3f sec' % ((time.time() - start),))
        self.dirty = False

    @staticmethod
    def _cache_key(context_tokens: List[int], model: RWKVModel, token: int) -> str:
        m = hashlib.sha1()

        m.update(model.args.model_id.encode('utf-8'))

        for context_token in context_tokens:
            m.update(struct.pack('i', context_token))

        m.update(struct.pack('i', token))

        return m.hexdigest()