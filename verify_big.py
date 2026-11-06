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
import os, pickle, sys, time, zlib
import numpy as np
import torch
import llz

cmd, src, limit, state_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
budget = float(sys.argv[5]) if len(sys.argv) > 5 else 240.0
MODEL = os.environ.get("LLZ_MODEL") or llz.DEFAULT_MODEL

raw_full = open(src, "rb").read()
raw = raw_full[:limit]
for _ in range(4):
    try:
        text = raw.decode("utf-8")
        break
    except UnicodeDecodeError:
        raw = raw[:-1]

tok, m, lib = llz.get_codec(MODEL)
lib.SCALE = 1 << 28
V = m.config.vocab_size
tokens = tok(text, return_tensors="pt").input_ids[0]
n = tokens.shape[0]
print(f"{src}: срез {len(raw)} байт, {n} токенов", flush=True)

if cmd == "encode":
    if os.path.exists(state_path):
        st = pickle.load(open(state_path, "rb"))
        enc, past, i0 = st["enc"], st["past"], st["i"]
        print(f"продолжаю с токена {i0}/{n}", flush=True)
    else:
        enc = lib.FastEncoder()
        uni_cum, uni_total = lib.quantize(np.full(V, 1.0 / V), V)
        enc.encode_idx(uni_cum, uni_total, int(tokens[0]))
        past, i0 = None, 0

    t0 = time.time()
    BLOCK = m.config.n_positions
    with torch.no_grad():
        for i in range(i0, n - 1):
            if past is not None and (i % BLOCK == 0):
                past = None
            cur = tokens[i].view(1, 1)
            row, past = lib.next_probs(m, cur, past)
            cum, total = lib.quantize(row, V)
            enc.encode_idx(cum, total, int(tokens[i + 1]))
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{n-1} ({(i+1-i0)/(time.time()-t0):.0f} ток/с)", flush=True)
            if time.time() - t0 > budget:
                break
        done_i = i + 1

    if done_i >= n - 1:
        payload = enc.finish()
        crc = zlib.crc32(raw) & 0xFFFFFFFF
        out = src + ".big.llz"
        open(out, "wb").write(llz.pack(payload, n, crc, MODEL, 16))
        if os.path.exists(state_path):
            os.remove(state_path)
        print(f"ГОТОВО: {len(raw)} байт -> {len(payload)} байт "
              f"({len(raw)/len(payload):.2f} : 1), записано: {out}", flush=True)
    else:
        pickle.dump({"enc": enc, "past": past, "i": done_i}, open(state_path, "wb"))
        print(f"чекпоинт: {done_i}/{n-1} ({done_i/(n-1)*100:.0f}%)", flush=True)

elif cmd == "decode":
    buf = open(src + ".big.llz", "rb").read()
    mid, scale, crc, n_dec, payload, _prof = llz.unpack(buf)
    assert n_dec == n, "число токенов в контейнере не совпало с текстом"

    if os.path.exists(state_path):
        st = pickle.load(open(state_path, "rb"))
        out_toks, past, i0 = st["out"], st["past"], st["i"]
        dec = lib.FastDecoder(payload)
        dec.lo, dec.hi, dec.code = st["lo"], st["hi"], st["code"]
        dec.reader.pos = st["pos"]
        print(f"продолжаю с токена {i0}/{n_dec}", flush=True)
    else:
        dec = lib.FastDecoder(payload)
        uni_cum, uni_total = lib.quantize(np.full(V, 1.0 / V), V)
        out_toks = [dec.decode_idx(uni_cum, uni_total)]
        past, i0 = None, 1

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
                print(f"  {i}/{n_dec} ({(i-i0)/(time.time()-t0):.0f} ток/с)", flush=True)
            if time.time() - t0 > budget:
                break

    if i >= n_dec:
        back = tok.decode(out_toks).encode("utf-8")
        open(src + ".big.out", "wb").write(back)
        if os.path.exists(state_path):
            os.remove(state_path)
        same = back == raw
        print(f"ГОТОВО: decode {n_dec} токенов", flush=True)
        print(f"ПОБАЙТОВОЕ СОВПАДЕНИЕ: {'ДА' if same else 'НЕТ!!!'}, "
              f"CRC32: {'OK' if (zlib.crc32(back) & 0xFFFFFFFF) == crc else 'НЕ СОШЁЛСЯ'}", flush=True)
        sys.exit(0 if same else 1)
    else:
        pickle.dump({"out": out_toks, "past": past, "i": i,
                     "lo": dec.lo, "hi": dec.hi, "code": dec.code,
                     "pos": dec.reader.pos}, open(state_path, "wb"))
        print(f"чекпоинт: {i}/{n_dec} ({i/n_dec*100:.0f}%)", flush=True)
