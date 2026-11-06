# -*- coding: utf-8 -*-
"""Кросс-машинный тест детерминизма: decode архива и сверка CRC32.
Запуск: python decode_test.py  (из папки cross_test)"""
import os, sys, zlib
os.environ["MKL_CBWR"] = "COMPATIBLE"      # MKL: одна ветка кода на любом CPU
os.environ["ATEN_CPU_CAPABILITY"] = "avx2"  # ATen: фиксированный SIMD-диспатч
os.environ["OMP_NUM_THREADS"] = "4"         # фикс. число потоков (редукции)
os.environ["LLZ_DTYPE"] = "fp64"   # портируемый профиль
os.environ["LLZ_DEVICE"] = "cpu"
sys.path.insert(0, ".")
import llz

import sys as _s
fname = _s.argv[1] if len(_s.argv) > 1 else "letter.cpu.llz"
buf = open(fname, "rb").read()
mid, scale, crc, n, payload, prof = llz.unpack(buf)
print(f"контейнер: {mid}, {n} токенов, профиль {prof}")
warn = llz.profile_mismatch(prof)
print(f"профиль: {'совпадает' if not warn else 'ОТЛИЧАЕТСЯ: ' + warn}")
text = llz.decode(payload, n, mid, scale)
raw = text.encode("utf-8")
ok = (zlib.crc32(raw) & 0xFFFFFFFF) == crc
print(f"CRC32: {'OK — КРОСС-МАШИННЫЙ ДЕТЕРМИНИЗМ ПОДТВЕРЖДЁН' if ok else 'НЕ СОШЁЛСЯ — fp-след разошёлся'}")
open(fname + ".decoded.txt", "wb").write(raw)
