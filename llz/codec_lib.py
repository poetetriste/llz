# -*- coding: utf-8 -*-
"""
Общая библиотека кодека: rugpt3small + арифметический кодер.
Кодирование и декодирование идут ОДИНАКОВЫМ пошаговым путём
(KV-cache, один токен за вызов), поэтому распределения совпадают
численно, а квантование 2^16 даёт запас от float-шума.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from .arith_coder import ArithmeticEncoder, ArithmeticDecoder

MODEL = "sberbank-ai/rugpt3small_based_on_gpt2"
PRECISION = 30            # точность арифм. кодера (биты); Python-int, переполнения нет
SCALE = 1 << (PRECISION - 2)  # total квантователя <= 2^(P-2) < мин. range кодера (2^(P-1)+1)

import os
# устройство: cuda если есть и не запрещено (LLZ_DEVICE=cpu принудительно)
DEVICE = os.environ.get("LLZ_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
# точность весов: fp32 эталон; LLZ_DTYPE=fp16 — быстрый профиль (свой fp-след)
_dt = os.environ.get("LLZ_DTYPE", "fp32")
DTYPE = {"fp16": torch.float16, "fp64": torch.float64}.get(_dt, torch.float32)
# LLZ_COMPILE=1 — torch.compile (inductor) поверх модели: ~2.6x быстрее,
# но inductor-фьюзии меняют fp-след => отдельный профиль контейнера.
# Требует MSVC cl.exe в PATH (запускать из vcvars64-окружения).
COMPILE = os.environ.get("LLZ_COMPILE") == "1"

_cache = {}
def get_model():
    if "m" not in _cache:
        tok = AutoTokenizer.from_pretrained(MODEL)
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE)
        m.eval()
        m.to(DEVICE)
        if COMPILE and DEVICE == "cuda":
            m = torch.compile(m)  # default mode, без cudagraphs
        _cache.update(tok=tok, m=m)
    return _cache["tok"], _cache["m"]

def quantize(p, V):
    assert SCALE > V, f"SCALE=2^{SCALE.bit_length()-1}={SCALE} должен быть > словаря V={V}"
    q = np.floor(p * (SCALE - V)).astype(np.int64) + 1
    cum = np.cumsum(q)
    return cum, int(cum[-1])

class FastEncoder(ArithmeticEncoder):
    def __init__(self):
        super().__init__(precision=PRECISION)
    def encode_idx(self, cum, total, idx):
        lo_c = int(cum[idx - 1]) if idx > 0 else 0
        rng = self.hi - self.lo + 1
        self.hi = self.lo + rng * int(cum[idx]) // total - 1
        self.lo = self.lo + rng * lo_c // total
        self._renorm()
    def _renorm(self):
        while True:
            if self.hi < self.HALF:
                self._emit(0)
            elif self.lo >= self.HALF:
                self._emit(1); self.lo -= self.HALF; self.hi -= self.HALF
            elif self.lo >= self.Q1 and self.hi < self.Q3:
                self.pending += 1; self.lo -= self.Q1; self.hi -= self.Q1
            else:
                break
            self.lo <<= 1
            self.hi = (self.hi << 1) | 1

class FastDecoder(ArithmeticDecoder):
    def __init__(self, data):
        super().__init__(data, precision=PRECISION)
    def decode_idx(self, cum, total):
        rng = self.hi - self.lo + 1
        val = ((self.code - self.lo + 1) * total - 1) // rng
        idx = int(np.searchsorted(cum, val + 1))
        lo_c = int(cum[idx - 1]) if idx > 0 else 0
        self.hi = self.lo + rng * int(cum[idx]) // total - 1
        self.lo = self.lo + rng * lo_c // total
        while True:
            if self.hi < self.HALF:
                pass
            elif self.lo >= self.HALF:
                self.lo -= self.HALF; self.hi -= self.HALF; self.code -= self.HALF
            elif self.lo >= self.Q1 and self.hi < self.Q3:
                self.lo -= self.Q1; self.hi -= self.Q1; self.code -= self.Q1
            else:
                break
            self.lo <<= 1
            self.hi = (self.hi << 1) | 1
            self.code = (self.code << 1) | self.reader.read()
        return idx

def crop_past(past, max_len):
    """Скользящее окно: оставить ПОСЛЕДНИЕ max_len позиций KV-cache.
    В transformers 5.x DynamicCache.crop(-N) удаляет НОВЕЙШИЕ токены
    (проверено эмпирически), поэтому режем тензоры слоёв напрямую."""
    if hasattr(past, "layers"):
        if past.get_seq_length() > max_len:
            for layer in past.layers:
                layer.keys = layer.keys[:, :, -max_len:, :]
                layer.values = layer.values[:, :, -max_len:, :]
        return past
    if hasattr(past, "key_cache"):      # transformers 4.x DynamicCache
        if past.get_seq_length() > max_len:
            past.key_cache = [k[:, :, -max_len:, :] for k in past.key_cache]
            past.value_cache = [v[:, :, -max_len:, :] for v in past.value_cache]
        return past
    return tuple((k[:, :, -max_len:, :], v[:, :, -max_len:, :]) for k, v in past)

def next_probs(m, cur, past):
    """Один шаг модели -> (np-вероятности, новый past).
    Контекст ограничен окном n_positions-1: без этого длинные тексты
    (>2048 токенов у rugpt3) падают в позиционном эмбеддинге."""
    W = m.config.n_positions - 1
    if past is not None:
        past = crop_past(past, W)
    res = m(input_ids=cur.to(DEVICE), past_key_values=past, use_cache=True)
    row = torch.softmax(res.logits[0, -1].float(), dim=-1).cpu().numpy()
    return row, res.past_key_values
