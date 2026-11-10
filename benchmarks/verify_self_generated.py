"""Reproduce the measured self-generated-text round trip."""

from __future__ import annotations

import bz2
import gzip
import hashlib
import json
import lzma
import time
import zlib
from pathlib import Path

import llz

SOURCE = Path(__file__).with_name("self_generated_degenerate.txt")
ARCHIVE = Path(__file__).with_name("self_generated_degenerate.llz3")
RESULT = Path(__file__).with_name("self_generated_result.json")
MODEL = "sberbank-ai/rugpt3small_based_on_gpt2"
EXPECTED_SHA256 = "adc66198463e24069c01406522ef072fe41c3735d9a7af6eb9dfe7e6dc30eed2"
ENTROPY_ESTIMATE_BITS = 920.7653650657554


def main() -> None:
    """Encode, pack, decode, verify, and record all measured sizes."""
    raw = SOURCE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"source hash mismatch: {digest}")
    text = raw.decode("utf-8")

    encode_started = time.time()
    payload, token_count = llz.encode(text, MODEL, 28)
    encode_seconds = time.time() - encode_started
    model = llz.get_codec(MODEL)[1]
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    container = llz.pack(
        payload,
        token_count,
        crc,
        MODEL,
        28,
        block=model.config.n_positions,
    )
    ARCHIVE.write_bytes(container)

    model_id, scale, expected_crc, tokens, packed_payload, profile = llz.unpack(
        container
    )
    decode_started = time.time()
    restored = llz.decode(
        packed_payload,
        tokens,
        model_id,
        scale,
        profile.get("model_revision"),
    ).encode("utf-8")
    decode_seconds = time.time() - decode_started
    byte_exact = restored == raw
    crc_ok = (zlib.crc32(restored) & 0xFFFFFFFF) == expected_crc
    if not byte_exact or not crc_ok:
        raise SystemExit("self-generated benchmark round trip failed")

    result = {
        "source_bytes": len(raw),
        "source_sha256": digest,
        "tokens": token_count,
        "model": MODEL,
        "model_revision": profile.get("model_revision"),
        "scale_bits": 28,
        "payload_bytes": len(payload),
        "payload_ratio": len(raw) / max(len(payload), 1),
        "llz3_container_bytes": len(container),
        "llz3_container_ratio": len(raw) / max(len(container), 1),
        "entropy_estimate_bits": ENTROPY_ESTIMATE_BITS,
        "entropy_estimate_ratio": len(raw) / (ENTROPY_ESTIMATE_BITS / 8),
        "gzip_bytes": len(gzip.compress(raw, compresslevel=9)),
        "bz2_bytes": len(bz2.compress(raw, compresslevel=9)),
        "xz_bytes": len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)),
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "byte_exact": byte_exact,
        "crc32_ok": crc_ok,
        "crc32": f"{crc:08x}",
        "profile": profile,
        "qualification": (
            "Artificial model-generated sample that collapsed into a repeated "
            "sentence; not representative of ordinary text."
        ),
    }
    RESULT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
