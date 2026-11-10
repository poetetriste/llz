# -*- coding: utf-8 -*-
"""
verify_corpora.py — честный полный круг encode->decode на реальных корпусах.

  python verify_corpora.py <file> <байт_среза> [модель]

Побайтовое сравнение результата с оригиналом. Без этого шага
любые цифры сжатия — теория, а не факт.
"""

import sys
import time
import zlib
import llz

src = sys.argv[1]
limit = int(sys.argv[2])
model = sys.argv[3] if len(sys.argv) > 3 else llz.DEFAULT_MODEL

raw_full = open(src, "rb").read()
raw = raw_full[:limit]
# режем по границе UTF-8: откусываем хвост, пока не станет валидным
for _ in range(4):
    try:
        text = raw.decode("utf-8")
        break
    except UnicodeDecodeError:
        raw = raw[:-1]
else:
    raise SystemExit("не удалось выровнять срез по UTF-8")

t0 = time.time()
blob, n = llz.encode(text, model, 28)
t_enc = time.time() - t0

t0 = time.time()
back = llz.decode(blob, n, model, 28).encode("utf-8")
t_dec = time.time() - t0

ok = back == raw
crc_ok = (zlib.crc32(back) & 0xFFFFFFFF) == (zlib.crc32(raw) & 0xFFFFFFFF)
ratio = len(raw) / max(1, len(blob))
print(f"\n=== {src} (срез {len(raw)} из {len(raw_full)} байт) ===")
print(f"токенов: {n}, сжато: {len(blob)} байт ({ratio:.2f} : 1)")
print(f"encode {t_enc:.0f} с ({n / max(1, t_enc):.1f} ток/с), decode {t_dec:.0f} с")
print(
    f"ПОБАЙТОВОЕ СОВПАДЕНИЕ: {'ДА' if ok else 'НЕТ!!!'}, CRC32: {'OK' if crc_ok else 'НЕ СОШЁЛСЯ'}"
)
sys.exit(0 if ok else 1)
