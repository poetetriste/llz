# -*- coding: utf-8 -*-
"""
test_llz.py — строгий стенд верификации кодека llz.

Часть A (без модели, быстро): стресс-тест арифметического кодера.
Часть B (с моделью): круг encode->decode на краевых случаях текста.

Запуск:
  python test_llz.py A          # только кодер (секунды)
  python test_llz.py B 0 6      # кейсы B с 0-го по 5-й
  python test_llz.py B 6 12
  python test_llz.py B 12 99    # остаток + детерминизм + контейнер
"""

import sys
import zlib
import numpy as np
from llz.errors import ConfigurationError, ContainerFormatError

PART = sys.argv[1] if len(sys.argv) > 1 else "A"
LO = int(sys.argv[2]) if len(sys.argv) > 2 else 0
HI = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9

FAILS = []


def check(name, cond, extra=""):
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}", flush=True)
    if extra:
        print(f"       {extra}", flush=True)
    if not cond:
        FAILS.append(name)


def part_a():
    print("=== A. Арифметический кодер (без модели) ===", flush=True)
    from llz import codec_lib
    from llz.codec_lib import FastEncoder, FastDecoder, quantize

    rng = np.random.default_rng(42)
    V = 50257

    ok = True
    for trial in range(200):
        p = rng.random(V)
        p /= p.sum()
        cum, total = quantize(p, V)
        k = int(rng.integers(1, 200))
        idxs = rng.integers(0, V, size=k)
        enc = FastEncoder()
        for i in idxs:
            enc.encode_idx(cum, total, int(i))
        dec = FastDecoder(enc.finish())
        if [dec.decode_idx(cum, total) for _ in range(k)] != [int(i) for i in idxs]:
            ok = False
            print(f"  trial {trial}: расхождение", flush=True)
            break
    check("A1: 200 раундов x до 200 символов, плотные p", ok)

    ok = True
    for trial in range(200):
        p = np.full(V, 1e-12)
        hot = int(rng.integers(0, V))
        p[hot] = 1.0
        p /= p.sum()
        cum, total = quantize(p, V)
        k = int(rng.integers(1, 100))
        idxs = [
            hot if rng.random() < 0.9 else int(rng.integers(0, V)) for _ in range(k)
        ]
        enc = FastEncoder()
        for i in idxs:
            enc.encode_idx(cum, total, i)
        dec = FastDecoder(enc.finish())
        if [dec.decode_idx(cum, total) for _ in range(k)] != idxs:
            ok = False
            break
    check("A2: вырожденные p (peak~1.0)", ok)

    ok = True
    for trial in range(50):
        p = rng.random(V)
        p /= p.sum()
        cum, total = quantize(p, V)
        idx = int(rng.integers(0, V))
        k = int(rng.integers(1, 2000))
        enc = FastEncoder()
        for _ in range(k):
            enc.encode_idx(cum, total, idx)
        dec = FastDecoder(enc.finish())
        if [dec.decode_idx(cum, total) for _ in range(k)] != [idx] * k:
            ok = False
            break
    check("A3: повтор одного символа до 2000 раз", ok)

    ok = True
    for sb in (
        16,
        20,
        24,
        28,
    ):  # валидные шкалы: total <= 2^28 < мин. range кодера 2^29
        codec_lib.SCALE = 1 << sb
        p = rng.random(V)
        p /= p.sum()
        cum, total = quantize(p, V)
        idxs = rng.integers(0, V, size=300)
        enc = FastEncoder()
        for i in idxs:
            enc.encode_idx(cum, total, int(i))
        dec = FastDecoder(enc.finish())
        if [dec.decode_idx(cum, total) for _ in range(300)] != [int(i) for i in idxs]:
            ok = False
            print(f"  scale 2^{sb} FAIL", flush=True)
    codec_lib.SCALE = 1 << 28
    check("A4: scale 2^16..2^28 (весь валидный диапазон)", ok)

    # A6: невалидный scale (< словаря) должен отвергаться, а не виснуть
    codec_lib.SCALE = 1 << 12
    try:
        quantize(np.full(V, 1.0 / V), V)
        check("A6: scale < V отвергается assertion'ом", False)
    except ConfigurationError:
        check("A6: scale < V отвергается явной ошибкой", True)
    finally:
        codec_lib.SCALE = 1 << 28

    enc = FastEncoder()
    dec = FastDecoder(enc.finish())
    check("A5: пустой поток", True)


CASES = [
    ("пустая строка", ""),
    ("один символ", "я"),
    ("одно слово", "Привет"),
    ("пробелы и переносы", "   \n\n\t  \r\n "),
    ("пунктуация", "!?.,;:-—…«»()[]{}\"'`~@#$%^&*+=<>/\\|"),
    ("кириллица+латиница+цифры", "Маша любит мяч 123 и mama с papa, но мяч меньше."),
    ("emoji и ZWJ", "Привет 👋🏽 семья 👨‍👩‍👧‍👦 флаги 🇷🇺🇺🇦 сердца ❤️‍🔥"),
    ("CJK", "これはテストです。这是一个测试。테스트입니다。"),
    ("арабский RTL", "هذا اختبار للنص العربي مع أرقام ١٢٣٤٥"),
    ("комбинирующая диакритика", "áёй ế ữ ố"),
    ("невидимые символы", "a​b‍c﻿d"),
    ("500 символов без пробелов", "а" * 500),
    ("повтор фразы x30", "Москва — столица России. " * 30),
    ("смесь скриптов", "Русский English 中文 العربية עברית Ελληνικά 日本語 한국어"),
    ("CRLF", "строка1\r\nстрока2\r\nстрока3"),
    ("табуляции", "a\tb\x0bc\x0cd"),
    ("похоже на код", "def f(x):\n    return [i**2 for i in range(x) if i % 2 == 0]\n"),
    ("числа и форматы", "π=3.1415926535, e=2.71828, 0xDEADBEEF, 1e-10, 1'000'000"),
]


def part_b():
    import llz

    MODEL = "sberbank-ai/rugpt3small_based_on_gpt2"

    for pos, (name, text) in enumerate(CASES):
        if not (LO <= pos < HI):
            continue
        try:
            raw = text.encode("utf-8")
            blob, n = llz.encode(text, MODEL, 28)
            back = llz.decode(blob, n, MODEL, 28)
            ok = back.encode("utf-8") == raw
            check(
                f"B{pos}: {name}",
                ok,
                f"{len(raw)} байт, {n} токенов -> {len(blob)} байт",
            )
        except Exception as e:
            check(f"B{pos}: {name}", False, f"исключение: {e!r}")

    if HI > len(CASES):  # финальный батч: детерминизм + контейнер
        text = "Детерминизм — основа воспроизводимости. Проверяем дважды."
        b1, n1 = llz.encode(text, MODEL, 28)
        b2, n2 = llz.encode(text, MODEL, 28)
        check(
            "B: детерминизм encode (2 прогона)",
            b1 == b2 and n1 == n2,
            f"{len(b1)} vs {len(b2)} байт",
        )

        raw = text.encode("utf-8")
        crc = zlib.crc32(raw) & 0xFFFFFFFF
        blob, n = llz.encode(text, MODEL, 28)
        bad = bytearray(blob)
        bad[len(bad) // 2] ^= 0xFF
        back = llz.decode(bytes(bad), n, MODEL, 28)
        crc_bad = zlib.crc32(back.encode("utf-8")) & 0xFFFFFFFF
        check("B: повреждённый бит -> CRC32 НЕ сходится", crc_bad != crc)

        packed = llz.pack(blob, n, crc, MODEL, 28)
        mid, sc, c2, n2, payload, _prof = llz.unpack(packed)
        check(
            "B: pack/unpack контейнера",
            mid == MODEL and sc == 28 and c2 == crc and n2 == n and payload == blob,
        )
        try:
            llz.unpack(b"XXXX" + packed[4:])
            check("B: чужой magic отвергается", False)
        except ContainerFormatError:
            check("B: чужой magic отвергается", True)

        # невалидный scale (2^30 > 2^28) должен отвергаться, а не виснуть
        try:
            llz.encode("тест", MODEL, 30)
            check("B: scale 2^30 отвергается", False)
        except ConfigurationError:
            check("B: scale 2^30 отвергается", True)


if PART == "A":
    part_a()
else:
    part_b()

print("\n" + "=" * 50, flush=True)
if FAILS:
    print(f"ИТОГ: {len(FAILS)} ПРОВАЛОВ: {FAILS}", flush=True)
    sys.exit(1)
print("ИТОГ: ВСЕ ТЕСТЫ ЗЕЛЁНЫЕ", flush=True)
