# -*- coding: utf-8 -*-
"""
verify_big.py — полный круг encode->decode на большом корпусе с чекпоинтами.

  python verify_big.py encode <file> <limit> <state.pkl> [бюджет_сек=240]
  python verify_big.py decode <file> <limit> <state.pkl> [бюджет_сек=240]

Каждый вызов обрабатывает столько токенов, сколько влезает в бюджет,
и сохраняет состояние (арифм. кодер + KV-cache + позиция) в state.pkl.
Повторные вызовы продолжают с места остановки. Когда прогресс = 100%,
encode пишет <file>.big.llz, decode пишет <file>.big.out и сверяет побайтово.
"""

import hashlib
import os
import pickle
import sys
import time
import zlib
from pathlib import Path
import numpy as np
import torch
import llz

cmd, src, limit, state_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
budget = float(sys.argv[5]) if len(sys.argv) > 5 else 240.0
MODEL = os.environ.get("LLZ_MODEL") or llz.DEFAULT_MODEL
SCALE_BITS_OVERRIDE = os.environ.get("LLZ_SCALE_BITS")
ENCODE_SCALE_BITS = int(SCALE_BITS_OVERRIDE or "28")

raw_full = open(src, "rb").read()
raw = raw_full[:limit]
for _ in range(4):
    try:
        text = raw.decode("utf-8")
        break
    except UnicodeDecodeError:
        raw = raw[:-1]

tok, m, lib = llz.get_codec(MODEL)
V = m.config.vocab_size
tokens = tok(text, return_tensors="pt").input_ids[0]
n = tokens.shape[0]
print(f"{src}: срез {len(raw)} байт, {n} токенов", flush=True)


def state_metadata(scale_bits):
    """Return identity metadata that prevents stale checkpoint reuse."""
    return {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "model": MODEL,
        "scale_bits": scale_bits,
        "tokens": int(n),
        "command": cmd,
    }


def load_state(scale_bits):
    """Load a trusted local checkpoint and validate its identity."""
    with open(state_path, "rb") as state_file:
        state = pickle.load(state_file)  # noqa: S301 - trusted local checkpoint only
    expected = state_metadata(scale_bits)
    if state.get("metadata") != expected:
        raise SystemExit(
            "чекпоинт относится к другому файлу/режиму; удалите его и начните заново"
        )
    return state


def save_state(state, scale_bits):
    """Atomically save a trusted local checkpoint."""
    state["metadata"] = state_metadata(scale_bits)
    target = Path(state_path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as state_file:
        pickle.dump(state, state_file)
    temporary.replace(target)


if cmd == "encode":
    lib.SCALE = 1 << ENCODE_SCALE_BITS
    if os.path.exists(state_path):
        st = load_state(ENCODE_SCALE_BITS)
        enc, past, i0 = st["enc"], st["past"], st["i"]
        print(f"продолжаю с токена {i0}/{n}", flush=True)
    else:
        enc = lib.FastEncoder()
        past, i0 = None, 0
        if n:
            uni_cum, uni_total = lib.quantize(np.full(V, 1.0 / V), V)
            enc.encode_idx(uni_cum, uni_total, int(tokens[0]))

    t0 = time.time()
    BLOCK = m.config.n_positions
    done_i = 0
    with torch.no_grad():
        for i in range(i0, n - 1):
            if past is not None and (i % BLOCK == 0):
                past = None
            cur = tokens[i].view(1, 1)
            row, past = lib.next_probs(m, cur, past)
            cum, total = lib.quantize(row, V)
            enc.encode_idx(cum, total, int(tokens[i + 1]))
            done_i = i + 1
            if (i + 1) % 500 == 0:
                print(
                    f"  {i + 1}/{n - 1} ({(i + 1 - i0) / (time.time() - t0):.0f} ток/с)",
                    flush=True,
                )
            if time.time() - t0 > budget:
                break

    if done_i >= n - 1:
        payload = enc.finish() if n else b""
        crc = zlib.crc32(raw) & 0xFFFFFFFF
        out = src + ".big.llz"
        with open(out, "wb") as output_file:
            output_file.write(llz.pack(payload, n, crc, MODEL, ENCODE_SCALE_BITS))
        if os.path.exists(state_path):
            os.remove(state_path)
        print(
            f"ГОТОВО: {len(raw)} байт -> {len(payload)} байт "
            f"({len(raw) / max(1, len(payload)):.2f} : 1), записано: {out}",
            flush=True,
        )
    else:
        save_state({"enc": enc, "past": past, "i": done_i}, ENCODE_SCALE_BITS)
        print(f"чекпоинт: {done_i}/{n - 1} ({done_i / (n - 1) * 100:.0f}%)", flush=True)

elif cmd == "decode":
    with open(src + ".big.llz", "rb") as archive_file:
        buf = archive_file.read()
    mid, scale, crc, n_dec, payload, _prof = llz.unpack(buf)
    if mid != MODEL:
        raise SystemExit(f"архив использует модель {mid}, а запрошена {MODEL}")
    if n_dec != n:
        raise SystemExit("число токенов в контейнере не совпало с текстом")
    decode_scale_bits = int(SCALE_BITS_OVERRIDE or scale)
    if SCALE_BITS_OVERRIDE and decode_scale_bits != scale:
        print(
            f"ВНИМАНИЕ: переопределяю ошибочный scale заголовка 2^{scale} "
            f"на 2^{decode_scale_bits}",
            flush=True,
        )
    lib.SCALE = 1 << decode_scale_bits

    if os.path.exists(state_path):
        st = load_state(decode_scale_bits)
        out_toks, past, i0 = st["out"], st["past"], st["i"]
        dec = lib.FastDecoder(payload)
        dec.lo, dec.hi, dec.code = st["lo"], st["hi"], st["code"]
        dec.reader.pos = st["pos"]
        print(f"продолжаю с токена {i0}/{n_dec}", flush=True)
    else:
        dec = lib.FastDecoder(payload)
        out_toks = []
        past, i0 = None, 0
        if n_dec:
            uni_cum, uni_total = lib.quantize(np.full(V, 1.0 / V), V)
            out_toks = [dec.decode_idx(uni_cum, uni_total)]
            i0 = 1

    t0 = time.time()
    BLOCK = m.config.n_positions
    with torch.no_grad():
        i = i0
        while i < n_dec:
            if past is not None and ((i - 1) % BLOCK == 0):
                past = None  # зеркально кодеру: сброс при подаче токена (i-1)%BLOCK==0
            cur = torch.tensor([[out_toks[-1]]])
            row, past = lib.next_probs(m, cur, past)
            cum, total = lib.quantize(row, V)
            out_toks.append(dec.decode_idx(cum, total))
            i += 1
            if i % 500 == 0:
                print(
                    f"  {i}/{n_dec} ({(i - i0) / (time.time() - t0):.0f} ток/с)",
                    flush=True,
                )
            if time.time() - t0 > budget:
                break

    if i >= n_dec:
        back = tok.decode(out_toks, clean_up_tokenization_spaces=False).encode("utf-8")
        with open(src + ".big.out", "wb") as output_file:
            output_file.write(back)
        if os.path.exists(state_path):
            os.remove(state_path)
        same = back == raw
        print(f"ГОТОВО: decode {n_dec} токенов", flush=True)
        print(
            f"ПОБАЙТОВОЕ СОВПАДЕНИЕ: {'ДА' if same else 'НЕТ!!!'}, "
            f"CRC32: {'OK' if (zlib.crc32(back) & 0xFFFFFFFF) == crc else 'НЕ СОШЁЛСЯ'}",
            flush=True,
        )
        sys.exit(0 if same else 1)
    else:
        save_state(
            {
                "out": out_toks,
                "past": past,
                "i": i,
                "lo": dec.lo,
                "hi": dec.hi,
                "code": dec.code,
                "pos": dec.reader.pos,
            },
            decode_scale_bits,
        )
        print(f"чекпоинт: {i}/{n_dec} ({i / n_dec * 100:.0f}%)", flush=True)
