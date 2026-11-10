"""Tests for the generic arithmetic coder implementation."""

from __future__ import annotations

import unittest

from llz.arith_coder import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    BitReader,
    BitWriter,
    to_cumfreq,
)


class BitStreamTests(unittest.TestCase):
    """Verify bit packing, padding, and reading."""

    def test_bit_writer_and_reader(self) -> None:
        """An incomplete final byte is padded and read in order."""
        bits = [1, 0, 1, 1, 0, 0, 1, 1, 1]
        writer = BitWriter()
        for bit in bits:
            writer.write(bit)

        packed = writer.to_bytes()
        reader = BitReader(packed)

        self.assertEqual(packed, bytes([0b10110011, 0b10000000]))
        self.assertEqual([reader.read() for _ in bits], bits)
        self.assertEqual([reader.read() for _ in range(7)], [0] * 7)
        self.assertEqual(reader.read(), 0)


class GenericArithmeticCoderTests(unittest.TestCase):
    """Verify the generic symbol-based coder, including renormalization."""

    def test_probability_conversion(self) -> None:
        """Cumulative frequencies are ordered and end at the requested scale."""
        cumulative, total = to_cumfreq({0: 0.1, 1: 0.2, 2: 0.7}, precision=12)

        self.assertEqual(total, 1 << 12)
        self.assertGreater(cumulative[0], 0)
        self.assertGreater(cumulative[1], cumulative[0])
        self.assertEqual(cumulative[2], total)

    def test_round_trip_exercises_all_renormalization_regions(self) -> None:
        """A skewed multi-symbol sequence round-trips exactly."""
        cumulative, total = to_cumfreq(
            {0: 0.01, 1: 0.09, 2: 0.2, 3: 0.7},
            precision=20,
        )
        message = ([3] * 30 + [0, 1, 2, 3, 2, 1, 0]) * 20
        encoder = ArithmeticEncoder(precision=20)
        for symbol in message:
            encoder.encode_symbol(cumulative, total, symbol)

        decoder = ArithmeticDecoder(encoder.finish(), precision=20)
        restored = [decoder.decode_symbol(cumulative, total) for _ in message]

        self.assertEqual(restored, message)


if __name__ == "__main__":
    unittest.main()
