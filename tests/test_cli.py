"""Model-free tests for the LLZ command-line boundary."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llz import cli


class CliTests(unittest.TestCase):
    """Verify CLI files, exit behavior, and integrity policy."""

    def _run(self, arguments: list[str]) -> str:
        """Run ``main`` with captured stdout."""
        output = io.StringIO()
        with patch("sys.argv", ["llz", *arguments]), contextlib.redirect_stdout(output):
            cli.main()
        return output.getvalue()

    @patch("llz.codec_lib.resolve_model_revision", return_value="c" * 40)
    def test_inspect_prints_llz3_metadata(self, _revision) -> None:
        """Inspect parses metadata without model inference."""
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "sample.llz"
            archive.write_bytes(cli.pack(b"abc", 2, 123, "org/model", 28))

            output = self._run(["inspect", str(archive)])

        self.assertIn("container: LLZ3", output)
        self.assertIn("model: org/model", output)
        self.assertIn("payload: 3 bytes", output)

    @patch("llz.cli.get_codec")
    @patch("llz.cli.encode", return_value=(b"encoded", 3))
    @patch("llz.codec_lib.resolve_model_revision", return_value="d" * 40)
    def test_encode_writes_container(
        self, _revision, encode_mock, get_codec_mock
    ) -> None:
        """Encode reports payload and full-container sizes separately."""
        get_codec_mock.return_value = (
            object(),
            SimpleNamespace(config=SimpleNamespace(n_positions=2_048)),
            object(),
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            archive = Path(directory) / "source.llz"
            source.write_text("Привет", encoding="utf-8")

            output = self._run(
                ["encode", str(source), "-o", str(archive), "--model", "org/model"]
            )
            restored = cli.unpack(archive.read_bytes())

        self.assertIn("container:", output)
        self.assertEqual(restored[0], "org/model")
        self.assertEqual(restored[3], 3)
        self.assertEqual(restored[4], b"encoded")
        encode_mock.assert_called_once()

    @patch("llz.cli.profile_mismatch", return_value="")
    @patch("llz.cli.decode", return_value="Привет")
    @patch("llz.codec_lib.resolve_model_revision", return_value=None)
    def test_decode_writes_only_crc_valid_output(
        self, _revision, decode_mock, _profile_mismatch
    ) -> None:
        """CRC-valid decoded bytes are written to the requested path."""
        raw = "Привет".encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "sample.llz"
            output_path = Path(directory) / "restored.txt"
            archive.write_bytes(
                cli.pack(b"payload", 2, zlib.crc32(raw), "org/model", 28)
            )

            output = self._run(["decode", str(archive), "-o", str(output_path)])

            self.assertEqual(output_path.read_bytes(), raw)
        self.assertIn("CRC32 OK", output)
        decode_mock.assert_called_once()

    @patch("llz.cli.profile_mismatch", return_value="")
    @patch("llz.cli.decode", return_value="wrong")
    @patch("llz.codec_lib.resolve_model_revision", return_value=None)
    def test_decode_refuses_crc_mismatch(
        self, _revision, _decode, _profile_mismatch
    ) -> None:
        """A CRC mismatch exits without creating an output file."""
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "sample.llz"
            output_path = Path(directory) / "restored.txt"
            archive.write_bytes(cli.pack(b"payload", 2, 1, "org/model", 28))

            with self.assertRaises(SystemExit):
                self._run(["decode", str(archive), "-o", str(output_path)])

            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
