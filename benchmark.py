"""Run byte-exact LLZ and classic-compressor benchmarks on one UTF-8 file."""

from __future__ import annotations

import argparse
import bz2
import gzip
import importlib.util
import lzma
import time
import zlib
from pathlib import Path

import llz


def _utf8_prefix(raw: bytes, limit: int | None) -> bytes:
    """Return a byte-limited prefix ending on a UTF-8 boundary."""
    data = raw if limit is None else raw[:limit]
    while data:
        try:
            data.decode("utf-8")
            return data
        except UnicodeDecodeError as error:
            if error.end != len(data):
                raise
            data = data[:-1]
    return b""


def main() -> None:
    """Run all configured compressors and print a Markdown result table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=llz.DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--scale", type=int, default=28)
    arguments = parser.parse_args()

    raw = _utf8_prefix(arguments.source.read_bytes(), arguments.limit)
    text = raw.decode("utf-8")
    print(f"file: {arguments.source} ({len(raw)} bytes)")

    rows: list[tuple[str, int, float]] = []

    def measure(name: str, compressor) -> None:
        started_at = time.time()
        compressed = compressor(raw)
        rows.append((name, len(compressed), time.time() - started_at))

    measure("gzip -9", lambda value: gzip.compress(value, compresslevel=9))
    if importlib.util.find_spec("brotli"):
        import brotli

        measure("brotli -q11", lambda value: brotli.compress(value, quality=11))
    measure(
        "xz -9e",
        lambda value: lzma.compress(value, preset=9 | lzma.PRESET_EXTREME),
    )
    measure("bz2 -9", lambda value: bz2.compress(value, compresslevel=9))

    print(f"llz: loading {arguments.model}", flush=True)
    started_at = time.time()
    payload, token_count = llz.encode(
        text,
        arguments.model,
        arguments.scale,
        arguments.revision,
    )
    encode_seconds = time.time() - started_at
    restored = llz.decode(
        payload,
        token_count,
        arguments.model,
        arguments.scale,
        arguments.revision,
    ).encode("utf-8")
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    if restored != raw or (zlib.crc32(restored) & 0xFFFFFFFF) != crc:
        raise SystemExit("LLZ round trip failed")
    container = llz.pack(
        payload,
        token_count,
        crc,
        arguments.model,
        arguments.scale,
        block=llz.get_codec(arguments.model, arguments.revision)[1].config.n_positions,
        model_revision=arguments.revision,
    )
    rows.append((f"llz payload ({arguments.model})", len(payload), encode_seconds))
    rows.append(("llz LLZ3 container", len(container), encode_seconds))

    print("\n| Method | Bytes | Ratio | Encode time |")
    print("|---|---:|---:|---:|")
    for name, size, seconds in sorted(rows, key=lambda row: row[1]):
        ratio = len(raw) / max(size, 1)
        print(f"| {name} | {size} | {ratio:.2f}:1 | {seconds:.2f} s |")
    print(f"\nLLZ round trip: byte-exact, CRC32 OK, {token_count} tokens")


if __name__ == "__main__":
    main()
