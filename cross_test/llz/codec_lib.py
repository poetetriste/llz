# -*- coding: utf-8 -*-
"""
Общая библиотека кодека: causal LM + арифметический кодер.
Кодирование и декодирование идут ОДИНАКОВЫМ пошаговым путём
(KV-cache, один токен за вызов), поэтому распределения совпадают
численно. Масштаб квантователя задаётся профилем архива; быстрый профиль
по умолчанию использует 2^28, переносимый CPU-профиль — 2^18.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .arith_coder import ArithmeticEncoder, ArithmeticDecoder
from .errors import ConfigurationError

MODEL = "sberbank-ai/rugpt3small_based_on_gpt2"
PRECISION = 30  # точность арифм. кодера (биты); Python-int, переполнения нет
SCALE = 1 << (
    PRECISION - 2
)  # total квантователя <= 2^(P-2) < мин. range кодера (2^(P-1)+1)

# устройство: cuda если есть и не запрещено (LLZ_DEVICE=cpu принудительно)
DEVICE = os.environ.get("LLZ_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu"
)
# точность весов: fp32 эталон; LLZ_DTYPE=fp16 — быстрый профиль (свой fp-след)
_dt = os.environ.get("LLZ_DTYPE", "fp32")
DTYPE = {"fp16": torch.float16, "fp64": torch.float64}.get(_dt, torch.float32)
# LLZ_COMPILE=1 — torch.compile (inductor) поверх модели: ~2.6x быстрее,
# но inductor-фьюзии меняют fp-след => отдельный профиль контейнера.
# Требует MSVC cl.exe в PATH (запускать из vcvars64-окружения).
COMPILE = os.environ.get("LLZ_COMPILE") == "1"

_cache: dict[tuple[str, str | None, torch.dtype, str, bool], tuple[object, object]] = {}


def get_model(model_id: str | None = None, revision: str | None = None):
    """Load and cache an exact model/profile combination.

    Args:
        model_id: Hugging Face model id or local model path.
        revision: Optional immutable Hub revision.

    Returns:
        Tokenizer and causal language model.
    """
    selected_model = model_id or MODEL
    key = (selected_model, revision, DTYPE, DEVICE, COMPILE)
    if key not in _cache:
        kwargs = {"revision": revision} if revision else {}
        tok = AutoTokenizer.from_pretrained(selected_model, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            selected_model,
            dtype=DTYPE,
            **kwargs,
        )
        model.eval()
        model.to(DEVICE)
        if COMPILE and DEVICE == "cuda":
            model = torch.compile(model)  # default mode, without CUDA graphs
        _cache[key] = (tok, model)
    return _cache[key]


def resolve_model_revision(model_id: str) -> str | None:
    """Resolve the immutable revision of a cached Hugging Face model.

    Local directories do not have a Hub revision and return ``None``.

    Args:
        model_id: Hugging Face model id or local model path.

    Returns:
        Cached commit hash when available.
    """
    configured = os.environ.get("LLZ_MODEL_REVISION")
    if configured:
        return configured
    if Path(model_id).exists():
        return None
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(model_id, local_files_only=True))
    except (OSError, ValueError):
        return None
    if snapshot.parent.name == "snapshots":
        return snapshot.name
    return None


def quantize(p: np.ndarray, V: int):
    """Convert probabilities to strictly positive integer frequencies.

    The returned total is at most ``SCALE``. Each vocabulary item receives
    one base count, while the remaining budget is assigned by flooring the
    scaled model probabilities.

    Args:
        p: One-dimensional probability vector.
        V: Vocabulary size.

    Raises:
        ConfigurationError: If the scale or probability vector is invalid.

    Returns:
        Cumulative frequencies and their integer total.
    """
    if SCALE <= V:
        raise ConfigurationError(
            f"SCALE={SCALE} must be greater than vocabulary size V={V}"
        )
    probabilities = np.asarray(p)
    if probabilities.shape != (V,):
        raise ConfigurationError(
            f"probability vector shape {probabilities.shape} does not match V={V}"
        )
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ConfigurationError("probabilities must be finite and non-negative")
    probability_sum = float(probabilities.sum())
    if probability_sum <= 0:
        raise ConfigurationError("probabilities must have a positive sum")
    q = np.floor(probabilities * (SCALE - V)).astype(np.int64) + 1
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
                self._emit(1)
                self.lo -= self.HALF
                self.hi -= self.HALF
            elif self.lo >= self.Q1 and self.hi < self.Q3:
                self.pending += 1
                self.lo -= self.Q1
                self.hi -= self.Q1
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
                self.lo -= self.HALF
                self.hi -= self.HALF
                self.code -= self.HALF
            elif self.lo >= self.Q1 and self.hi < self.Q3:
                self.lo -= self.Q1
                self.hi -= self.Q1
                self.code -= self.Q1
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
    if hasattr(past, "key_cache"):  # transformers 4.x DynamicCache
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
