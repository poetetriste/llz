"""Integrity tests for committed benchmark evidence."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import llz

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


class BenchmarkArtifactTests(unittest.TestCase):
    """Verify committed corpora and archives match their manifest."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the benchmark manifest once."""
        cls.manifest = json.loads(
            (BENCHMARKS / "manifest.json").read_text(encoding="utf-8")
        )

    def _assert_file(self, section: dict[str, object], path_key: str = "path") -> bytes:
        """Assert a manifest section's size and SHA-256."""
        path = BENCHMARKS / str(section[path_key])
        raw = path.read_bytes()
        self.assertEqual(len(raw), section["bytes"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), section["sha256"])
        return raw

    def test_wikipedia_corpus(self) -> None:
        """The headline corpus is byte-identical to the documented input."""
        self._assert_file(self.manifest["wikipedia_corpus"])

    def test_repaired_main_archive_metadata(self) -> None:
        """The control LLZ2 records the encoder's actual scale and payload."""
        section = self.manifest["main_llz2_result"]
        raw = self._assert_file(section)
        model, scale, crc, tokens, payload, profile = llz.unpack(raw)

        self.assertEqual(model, section["model"])
        self.assertEqual(scale, section["scale_bits"])
        self.assertEqual(f"{crc:08x}", section["crc32"])
        self.assertEqual(tokens, section["tokens"])
        self.assertEqual(len(payload), section["payload_bytes"])
        self.assertEqual(profile["container_version"], 2)

    def test_self_generated_evidence(self) -> None:
        """The self-generated corpus and LLZ3 match all recorded hashes."""
        section = self.manifest["self_generated_degenerate"]
        self._assert_file(section)
        archive_section = section["llz3_container"]
        raw = self._assert_file(archive_section)
        model, scale, crc, tokens, payload, profile = llz.unpack(raw)

        self.assertEqual(model, section["model"])
        self.assertEqual(scale, 28)
        self.assertEqual(tokens, section["tokens"])
        self.assertEqual(len(payload), section["measured_arithmetic_payload"]["bytes"])
        self.assertEqual(f"{crc:08x}", archive_section["crc32"])
        self.assertEqual(profile["model_revision"], archive_section["model_revision"])
        self.assertEqual(profile["container_version"], 3)


if __name__ == "__main__":
    unittest.main()
