# -*- coding: utf-8 -*-
"""
Арифметический кодер с настраиваемой точностью и битами подтверждения.
Модель снаружи: для каждой позиции отдаёт распределение вероятностей
по алфавиту, кодер превращает его в битовый поток.
"""

from .errors import ContainerFormatError


class BitWriter:
    def __init__(self):
        self.bits = []

    def write(self, bit):
        self.bits.append(bit)

    def to_bytes(self):
        out = bytearray()
        for i in range(0, len(self.bits), 8):
            chunk = self.bits[i : i + 8]
            while len(chunk) < 8:
                chunk.append(0)
            out.append(int("".join(map(str, chunk)), 2))
        return bytes(out)


class BitReader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read(self):
        byte = self.data[self.pos // 8] if self.pos // 8 < len(self.data) else 0
        bit = (byte >> (7 - self.pos % 8)) & 1
        self.pos += 1
        return bit


def to_cumfreq(probs, precision=20):
    """Вероятности -> целочисленные накопленные частоты (сумма = 2^precision)."""
    scale = 1 << precision
    cum, total, prev = {}, 0, 0.0
    items = sorted(probs.items())
    for sym, p in items:
        prev += p
        # гарантируем минимум 1 и монотонность
        cum[sym] = max(total + 1, round(prev * scale))
        total = cum[sym]
    total = scale
    return cum, total


class ArithmeticEncoder:
    def __init__(self, precision=20):
        self.P = precision
        self.HALF = 1 << (precision - 1)
        self.Q1 = self.HALF >> 1
        self.Q3 = self.Q1 * 3
        self.lo = 0
        self.hi = (1 << precision) - 1
        self.pending = 0
        self.out = BitWriter()

    def _emit(self, bit):
        self.out.write(bit)
        for _ in range(self.pending):
            self.out.write(bit ^ 1)
        self.pending = 0

    def encode_symbol(self, cum, total, sym):
        rng = self.hi - self.lo + 1
        syms = sorted(cum)
        idx = syms.index(sym)
        lo_c = cum[syms[idx - 1]] if idx > 0 else 0
        hi_c = cum[sym] - 1
        self.hi = self.lo + rng * (hi_c + 1) // total - 1
        self.lo = self.lo + rng * lo_c // total
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

    def finish(self):
        self.pending += 1
        self._emit(1 if self.lo >= self.Q1 else 0)
        return self.out.to_bytes()


class ArithmeticDecoder:
    def __init__(self, data, precision=20):
        self.P = precision
        self.HALF = 1 << (precision - 1)
        self.Q1 = self.HALF >> 1
        self.Q3 = self.Q1 * 3
        self.lo = 0
        self.hi = (1 << precision) - 1
        self.reader = BitReader(data)
        self.code = 0
        for _ in range(precision):
            self.code = (self.code << 1) | self.reader.read()

    def decode_symbol(self, cum, total):
        rng = self.hi - self.lo + 1
        offset = self.code - self.lo
        target = (offset + 1) * total - 1
        val = target // rng
        syms = sorted(cum)
        sym, lo_c = None, 0
        for s in syms:
            if cum[s] - 1 >= val >= lo_c:
                sym = s
                break
            lo_c = cum[s]
        if sym is None:
            raise ContainerFormatError("arithmetic decoder desynchronized")
        self.hi = self.lo + rng * cum[sym] // total - 1
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
        return sym


if __name__ == "__main__":
    import random

    random.seed(1)
    alphabet = list(range(256))
    # Deterministic non-cryptographic data for the module self-test.
    weights = {s: random.random() ** 3 for s in alphabet}  # noqa: S311
    sw = sum(weights.values())
    probs = {s: w / sw for s, w in weights.items()}
    cum, total = to_cumfreq(probs)

    message = [
        random.choices(alphabet, weights=[probs[s] for s in alphabet])[0]  # noqa: S311
        for _ in range(10000)
    ]

    enc = ArithmeticEncoder()
    for s in message:
        enc.encode_symbol(cum, total, s)
    blob = enc.finish()

    dec = ArithmeticDecoder(blob)
    decoded = [dec.decode_symbol(cum, total) for _ in range(len(message))]

    if decoded != message:
        raise RuntimeError("arithmetic coder round trip failed")
    import math

    ideal = -sum(math.log2(probs[s]) for s in message) / 8
    print(
        f"round-trip OK: {len(message)} символов -> {len(blob)} байт "
        f"(идеал по энтропии {ideal:.0f} байт, overhead {len(blob) / ideal - 1:+.2%})"
    )
