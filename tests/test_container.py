"""Container-format regression tests."""

from __future__ import annotations

import struct
import unittest
from unittest.mock import patch

from llz import cli
from llz.errors import ContainerFormatError


class ContainerTests(unittest.TestCase):
    """Verify LLZ container compatibility and validation."""

    @patch("llz.codec_lib.resolve_model_revision", return_value="a" * 40)
    def test_llz3_round_trip(self, _revision) -> None:
        """LLZ3 metadata and payload survive pack/unpack exactly."""
        payload = b"payload-bytes"
        container = cli.pack(payload, 123, 0xDEADBEEF, "org/model", 28)
        model, scale, crc, tokens, restored, profile = cli.unpack(container)

        self.assertEqual(model, "org/model")
        self.assertEqual(scale, 28)
        self.assertEqual(crc, 0xDEADBEEF)
        self.assertEqual(tokens, 123)
        self.assertEqual(restored, payload)
        self.assertEqual(profile["container_version"], 3)
        self.assertEqual(profile["model_revision"], "a" * 40)
        self.assertEqual(profile["payload_size"], len(payload))

    def test_llz2_backward_compatibility(self) -> None:
        """Existing LLZ2 archives remain parseable."""
        model = b"org/model"
        payload = b"legacy"
        container = (
            cli.MAGIC2
            + struct.pack("<B", len(model))
            + model
            + struct.pack("<IBBI", 18, 2, 0, 2_048)
            + struct.pack("<IQ", 1234, 17)
            + payload
        )

        restored = cli.unpack(container)

        self.assertEqual(restored[:4], ("org/model", 18, 1234, 17))
        self.assertEqual(restored[4], payload)
        self.assertEqual(restored[5]["container_version"], 2)

    @patch("llz.codec_lib.resolve_model_revision", return_value=None)
    def test_payload_size_mismatch_is_rejected(self, _revision) -> None:
        """Truncated and extended LLZ3 payloads are rejected."""
        container = cli.pack(b"abc", 1, 0, "org/model", 28)
        with self.assertRaises(ContainerFormatError):
            cli.unpack(container[:-1])
        with self.assertRaises(ContainerFormatError):
            cli.unpack(container + b"x")

    def test_all_truncated_prefixes_are_rejected(self) -> None:
        """No truncated LLZ3 header is accepted or crashes unexpectedly."""
        with patch("llz.codec_lib.resolve_model_revision", return_value="b" * 40):
            container = cli.pack(b"abc", 2, 42, "org/model", 28)
        for size in range(4 + cli._LLZ3_FIXED.size):
            with self.subTest(size=size):
                with self.assertRaises(ContainerFormatError):
                    cli.unpack(container[:size])

    def test_unknown_magic_is_rejected(self) -> None:
        """Foreign files are rejected with a format error."""
        with self.assertRaises(ContainerFormatError):
            cli.unpack(b"NOPE-not-an-archive")

    @patch("llz.codec_lib.resolve_model_revision", return_value=None)
    def test_pack_rejects_out_of_range_fields(self, _revision) -> None:
        """Invalid values cannot be serialized into ambiguous LLZ3 headers."""
        invalid_calls = (
            lambda: cli.pack(bytearray(b"x"), 1, 0, "model", 28),
            lambda: cli.pack(b"x", -1, 0, "model", 28),
            lambda: cli.pack(b"x", 1, -1, "model", 28),
            lambda: cli.pack(b"x", 1, 0, "model", 28, block=0),
            lambda: cli.pack(b"x", 1, 0, "model", 0),
            lambda: cli.pack(b"x", 1, 0, "", 28),
            lambda: cli.pack(b"x", 1, 0, "model", 28, model_revision="r" * 256),
            lambda: cli.pack(b"x", 1, 0, "model", 28, software_profile="s" * 256),
        )
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises(ContainerFormatError):
                    invalid_call()

    @patch("llz.codec_lib.resolve_model_revision", return_value=None)
    def test_invalid_profile_flags_are_rejected(self, _revision) -> None:
        """Impossible dtype/device metadata cannot reach model loading."""
        container = bytearray(cli.pack(b"abc", 1, 0, "org/model", 28))
        dtype_offset = 4 + 2 + 1 + 1 + 1
        container[dtype_offset] = 99
        with self.assertRaises(ContainerFormatError):
            cli.unpack(bytes(container))


if __name__ == "__main__":
    unittest.main()
