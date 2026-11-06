# -*- coding: utf-8 -*-
"""
benchmark.py — llz против классики на одном файле.

  python benchmark.py corpus.txt [лимит_байт]

Печатает markdown-таблицу. llz прогоняет полный круг (encode→decode→CRC32),
классика — только сжатие (их корректность гарантирует библиотека).
"""
import bz2, gzip, lzma, sys, time, zlib

import llz

src = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
raw = open(src, "rb").read()
if limit:
    raw = raw[:limit]
print(f"файл: {src} ({len(raw)} байт)")

rows = []
def row(name, size, sec=None):
    rows.append((name, size, len(raw) / size if size else 0.0, sec))

import importlib.util
t0 = time.time(); row("gzip -9", len(gzip.compress(raw, 9)), time.time() - t0)
if importlib.util.find_spec("brotli"):
    t0 = time.time()
    row("brotli -q11", len(__import__("brotli").compress(raw, quality=11)), time.time() - t0)
t0 = time.time(); row("xz -9e", len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)), time.time() - t0)
t0 = time.time(); row("bz2 -9", len(bz2.compress(raw, 9)), time.time() - t0)

model = llz.DEFAULT_MODEL
print(f"llz: модель {model}, кодирую...", flush=True)
text = raw.decode("utf-8")
t0 = time.time()
blob, n = llz.encode(text, model, 28)
t_enc = time.time() - t0
back = llz.decode(blob, n, model, 28).encode("utf-8")
ok = back == raw and (zlib.crc32(back) & 0xFFFFFFFF) == (zlib.crc32(raw) & 0xFFFFFFFF)
row(f"llz ({model.split('/')[-1]})", len(blob), t_enc)
print(f"llz: круг {'OK, побайтово' if ok else 'СЛОМАН!'} "
      f"(encode {t_enc:.0f} с, {n} токенов)", flush=True)

print("\n| Метод | Размер | Степень |")
print("|---|---|---|")
for name, size, ratio, sec in sorted(rows, key=lambda r: r[1]):
    if size:
        print(f"| {name} | {size} | {ratio:.2f} : 1 |")
