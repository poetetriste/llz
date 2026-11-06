# -*- coding: utf-8 -*-
"""
llz — LLM-компрессор текста. CLI-инструмент.

  python llz.py encode book.txt [-o book.llz]
  python llz.py decode book.llz [-o book.out.txt]
  python llz.py bench  book.txt          # только замер, без записи

Контейнер .llz (бинарный):
  magic "LLZ1" | model_id_len u8 | model_id | scale u32 |
  n_tokens u64 | crc32 u32 | payload (арифм. поток)
Декодер работает только с той же моделью, что и кодер.
"""
import argparse, json, struct, sys, time, zlib
import numpy as np
import torch

MAGIC = b"LLZ1"
MAGIC2 = b"LLZ2"
DEFAULT_MODEL = "sberbank-ai/rugpt3small_based_on_gpt2"

_state = {}
def get_codec(model_id):
    if _state.get("id") != model_id:
        from . import codec_lib
        codec_lib.MODEL = model_id
        tok, m = codec_lib.get_model()
        _state.update(id=model_id, tok=tok, m=m, lib=codec_lib)
    return _state["tok"], _state["m"], _state["lib"]

def encode(text: str, model_id: str, scale_bits: int):
    tok, m, lib = get_codec(model_id)
    V = m.config.vocab_size
    # инварианты: SCALE > V (у каждого токена счётчик >= 1)
    # и SCALE <= 2^18 (ренормализация кодера держит range > 2^18;
    # если total > range, интервал инвертируется и кодер зависает)
    assert (1 << scale_bits) > V, f"scale 2^{scale_bits} <= словаря V={V}"
    assert scale_bits <= lib.PRECISION - 2, \
        f"scale 2^{scale_bits} > 2^{lib.PRECISION-2} — кодер не гарантирует range > total"
    lib.SCALE = 1 << scale_bits
    BLOCK = m.config.n_positions
    tokens = tok(text, return_tensors="pt").input_ids[0]
    n = tokens.shape[0]
    if n == 0:
        return b"", 0
    enc = lib.FastEncoder()
    uni_cum, uni_total = lib.quantize(np.full(V, 1.0 / V), V)
    enc.encode_idx(uni_cum, uni_total, int(tokens[0]))
    past, cur = None, tokens[0].view(1, 1)
    t0 = time.time()
    with torch.no_grad():
        for i in range(n - 1):
            if past is not None and (i % BLOCK == 0):
                past = None  # сброс блока: честные позиции вместо скользящего окна
            row, past = lib.next_probs(m, cur, past)
            cum, total = lib.quantize(row, V)
            enc.encode_idx(cum, total, int(tokens[i + 1]))
            cur = tokens[i + 1].view(1, 1)
            if (i + 1) % 2000 == 0:
                print(f"  encode {i+1}/{n-1} ({(i+1)/(time.time()-t0):.0f} ток/с)",
                  flush=True)
    return enc.finish(), n

def encode_batched(text: str, model_id: str, scale_bits: int):
    """Быстрый encode: один forward на блок BLOCK токенов (teacher forcing)
    вместо пошагового KV-cache. Битовый поток идентичен схеме с блочным
    сбросом в точной арифметике; на практике возможен fp-дрейф между
    батчевым и пошаговым ядрами — поэтому результат ОБЯЗАТЕЛЬНО
    верифицируется декодированием (CRC32) на целевом профиле."""
    tok, m, lib = get_codec(model_id)
    V = m.config.vocab_size
    assert (1 << scale_bits) > V, f"scale 2^{scale_bits} <= словаря V={V}"
    assert scale_bits <= lib.PRECISION - 2
    lib.SCALE = 1 << scale_bits
    BLOCK = m.config.n_positions
    tokens = tok(text, return_tensors="pt").input_ids[0]
    n = tokens.shape[0]
    if n == 0:
        return b"", 0
    enc = lib.FastEncoder()
    uni_cum, uni_total = lib.quantize(np.full(V, 1.0 / V), V)
    enc.encode_idx(uni_cum, uni_total, int(tokens[0]))
    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, n - 1, BLOCK):
            seg = tokens[b0:b0 + BLOCK].to(lib.DEVICE).unsqueeze(0)
            logits = m(seg).logits[0].float()          # [L, V]
            L = logits.shape[0]
            # logits[j] предсказывает tokens[b0+j+1]. Если это НЕ хвостовой
            # блок, последняя строка кодирует первый токен следующего блока
            # (зеркально схеме сброса в пошаговом пути).
            k = L if b0 + L < n else L - 1
            probs = torch.softmax(logits[:k], dim=-1).cpu().numpy()
            for j in range(k):
                cum, total = lib.quantize(probs[j], V)
                enc.encode_idx(cum, total, int(tokens[b0 + j + 1]))
            done = min(b0 + k, n - 1)
            print(f"  encode {done}/{n-1} ({done/max(1e-9,time.time()-t0):.0f} ток/с)",
                  flush=True)
    return enc.finish(), n

def decode(payload: bytes, n: int, model_id: str, scale_bits: int) -> str:
    tok, m, lib = get_codec(model_id)
    V = m.config.vocab_size
    assert (1 << scale_bits) > V, f"scale 2^{scale_bits} <= словаря V={V}"
    assert scale_bits <= lib.PRECISION - 2, \
        f"scale 2^{scale_bits} > 2^{lib.PRECISION-2} — кодер не гарантирует range > total"
    lib.SCALE = 1 << scale_bits
    BLOCK = m.config.n_positions
    dec = lib.FastDecoder(payload)
    uni_cum, uni_total = lib.quantize(np.full(V, 1.0 / V), V)
    if n == 0:
        return ""
    out = [dec.decode_idx(uni_cum, uni_total)]
    past, cur = None, torch.tensor([[out[0]]])
    t0 = time.time()
    with torch.no_grad():
        for i in range(n - 1):
            if past is not None and (i % BLOCK == 0):
                past = None  # сброс блока, зеркально кодеру
            row, past = lib.next_probs(m, cur, past)
            cum, total = lib.quantize(row, V)
            out.append(dec.decode_idx(cum, total))
            cur = torch.tensor([[out[-1]]])
            if (i + 1) % 2000 == 0:
                print(f"  decode {i+1}/{n-1} ({(i+1)/(time.time()-t0):.0f} ток/с)",
                  flush=True)
    return tok.decode(out)

def pack(payload, n, crc, model_id, scale_bits, block=2048):
    """Контейнер LLZ2: magic | model_id | scale u32 | dtype u8 | device u8 |
    block u32 | crc32 u32 | n_tokens u64 | payload.
    dtype: 0=fp32, 1=fp16, 2=fp64. device: 0=cpu, 1=cuda, 2=cuda+torch.compile.
    Профиль (dtype+device) обязателен: fp-ядра разных устройств дают разные
    распределения, арифметический кодер требует битовой идентичности."""
    from . import codec_lib
    dtype_flag = {torch.float32: 0, torch.float16: 1, torch.float64: 2}[codec_lib.DTYPE]
    device_flag = 1 if codec_lib.DEVICE == "cuda" else 0
    if device_flag == 1 and codec_lib.COMPILE:
        device_flag = 2
    mid = model_id.encode()
    return (MAGIC2 + struct.pack("<B", len(mid)) + mid
            + struct.pack("<IBBI", scale_bits, dtype_flag, device_flag, block)
            + struct.pack("<IQ", crc, n) + payload)

def unpack(buf):
    if buf[:4] == MAGIC2:
        mlen = buf[4]
        mid = buf[5:5+mlen].decode()
        scale, dt, dev, block = struct.unpack("<IBBI", buf[5+mlen:5+mlen+10])
        crc, n = struct.unpack("<IQ", buf[5+mlen+10:5+mlen+22])
        return mid, scale, crc, n, buf[5+mlen+22:], {"dtype": dt, "device": dev, "block": block}
    # legacy LLZ1: fp32, cpu, block 2048
    assert buf[:4] == MAGIC, "не LLZ1/LLZ2-файл"
    mlen = buf[4]
    mid = buf[5:5+mlen].decode()
    scale, crc, n = struct.unpack("<IIQ", buf[5+mlen:5+mlen+16])
    return mid, scale, crc, n, buf[5+mlen+16:], {"dtype": 0, "device": 0, "block": 2048}

def profile_mismatch(prof):
    """Предупреждение, если архив писался под другой fp-профиль."""
    from . import codec_lib
    want_dt = {torch.float32: 0, torch.float16: 1, torch.float64: 2}[codec_lib.DTYPE]
    want_dev = 1 if codec_lib.DEVICE == "cuda" else 0
    if want_dev == 1 and codec_lib.COMPILE:
        want_dev = 2
    msgs = []
    if prof["dtype"] != want_dt:
        msgs.append(f"dtype архива={prof['dtype']} vs сейчас={want_dt}")
    if prof["device"] != want_dev:
        msgs.append(f"device архива={prof['device']} vs сейчас={want_dev}")
    return "; ".join(msgs)

def main():
    ap = argparse.ArgumentParser(prog="llz")
    ap.add_argument("cmd", choices=["encode", "decode", "bench"])
    ap.add_argument("src")
    ap.add_argument("-o", "--out")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--scale", type=int, default=28)
    ap.add_argument("--batched", action="store_true",
                    help="быстрый encode через teacher forcing (требует CRC-проверки)")
    a = ap.parse_args()

    if a.cmd in ("encode", "bench"):
        raw = open(a.src, "rb").read()
        text = raw.decode("utf-8")
        t0 = time.time()
        blob, n = (encode_batched if a.batched else encode)(text, a.model, a.scale)
        dt = time.time() - t0
        crc = zlib.crc32(raw) & 0xFFFFFFFF
        print(f"\n{len(raw)} байт -> {len(blob)} байт "
              f"({len(raw)/len(blob):.2f} : 1), {n} токенов, {dt:.0f} с")
        import lzma, bz2
        z, b = lzma.compress(raw, preset=9), bz2.compress(raw, 9)
        print(f"  xz: {len(z)} ({len(raw)/len(z):.2f} : 1), "
              f"bz2: {len(b)} ({len(raw)/len(b):.2f} : 1)")
        if a.cmd == "encode":
            out = a.out or a.src.rsplit(".", 1)[0] + ".llz"
            open(out, "wb").write(pack(blob, n, crc, a.model, a.scale))
            print(f"записано: {out}")
    else:
        buf = open(a.src, "rb").read()
        mid, scale, crc, n, payload, prof = unpack(buf)
        warn = profile_mismatch(prof)
        print(f"контейнер: модель {mid}, {n} токенов, {len(payload)} байт данных, "
              f"профиль dtype={prof['dtype']} device={prof['device']} block={prof['block']}")
        if warn:
            print(f"ВНИМАНИЕ: профиль не совпадает с текущим ({warn}) — "
                  f"CRC32 рассудит, но лучше декодировать на профиле записи")
        text = decode(payload, n, mid, scale)
        raw = text.encode("utf-8")
        ok = (zlib.crc32(raw) & 0xFFFFFFFF) == crc
        print(f"CRC32 {'OK — восстановлено побайтово' if ok else 'НЕ СОШЁЛСЯ!'}")
        out = a.out or a.src.rsplit(".", 1)[0] + ".decoded.txt"
        open(out, "wb").write(raw)
        print(f"записано: {out}")

if __name__ == "__main__":
    main()
