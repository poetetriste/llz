"""Command-line interface and public container API for LLZ."""

from __future__ import annotations

import argparse
import bz2
import lzma
import platform
import struct
import time
import zlib
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import transformers

from .errors import ConfigurationError, ContainerFormatError

MAGIC = b"LLZ1"
MAGIC2 = b"LLZ2"
MAGIC3 = b"LLZ3"
DEFAULT_MODEL = "sberbank-ai/rugpt3small_based_on_gpt2"

_LLZ3_FIXED = struct.Struct("<HBBBBBIIQQ")
_LLZ2_PROFILE = struct.Struct("<IBBI")
_LLZ2_TAIL = struct.Struct("<IQ")
_LLZ1_TAIL = struct.Struct("<IIQ")
_MAX_MODEL_ID_BYTES = 65_535
_MAX_REVISION_BYTES = 255


class Profile(TypedDict, total=False):
    """Compute and provenance metadata stored in a container."""

    dtype: int
    device: int
    block: int
    model_revision: str | None
    container_version: int
    payload_size: int
    software_profile: str


_state: dict[tuple[str, str | None], tuple[object, object, object]] = {}


def get_codec(model_id: str, revision: str | None = None):
    """Return a cached tokenizer, model, and codec module."""
    key = (model_id, revision)
    if key not in _state:
        from . import codec_lib

        tokenizer, model = codec_lib.get_model(model_id, revision)
        _state[key] = (tokenizer, model, codec_lib)
        if revision is None:
            resolved_revision = codec_lib.resolve_model_revision(model_id)
            if resolved_revision:
                _state[(model_id, resolved_revision)] = _state[key]
    return _state[key]


def _validate_scale(scale_bits: int, vocabulary_size: int, precision: int) -> None:
    """Validate a quantizer scale against codec invariants."""
    if not isinstance(scale_bits, int):
        raise ConfigurationError("scale_bits must be an integer")
    if scale_bits < 1 or scale_bits > precision - 2:
        raise ConfigurationError(
            f"scale_bits must be between 1 and {precision - 2}, got {scale_bits}"
        )
    if (1 << scale_bits) <= vocabulary_size:
        raise ConfigurationError(
            f"scale 2^{scale_bits} must exceed vocabulary size {vocabulary_size}"
        )


def _current_profile() -> tuple[int, int]:
    """Return dtype and device flags for the active compute profile."""
    from . import codec_lib

    dtype_flag = {
        torch.float32: 0,
        torch.float16: 1,
        torch.float64: 2,
    }.get(codec_lib.DTYPE)
    if dtype_flag is None:
        raise ConfigurationError(f"unsupported dtype: {codec_lib.DTYPE}")
    device_flag = 1 if str(codec_lib.DEVICE).startswith("cuda") else 0
    if device_flag == 1 and codec_lib.COMPILE:
        device_flag = 2
    return dtype_flag, device_flag


def _software_profile() -> str:
    """Return the software versions that affect floating-point predictions."""
    return (
        f"python={platform.python_version()};torch={torch.__version__};"
        f"transformers={transformers.__version__};numpy={np.__version__}"
    )


def encode(
    text: str,
    model_id: str,
    scale_bits: int,
    revision: str | None = None,
) -> tuple[bytes, int]:
    """Encode UTF-8 text into an arithmetic-coded payload."""
    tokenizer, model, lib = get_codec(model_id, revision)
    vocabulary_size = model.config.vocab_size
    _validate_scale(scale_bits, vocabulary_size, lib.PRECISION)
    lib.SCALE = 1 << scale_bits
    block_size = model.config.n_positions
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    token_count = int(tokens.shape[0])
    if token_count == 0:
        return b"", 0

    encoder = lib.FastEncoder()
    uniform_cumulative, uniform_total = lib.quantize(
        np.full(vocabulary_size, 1.0 / vocabulary_size), vocabulary_size
    )
    encoder.encode_idx(uniform_cumulative, uniform_total, int(tokens[0]))
    past = None
    current = tokens[0].view(1, 1)
    started_at = time.time()
    with torch.no_grad():
        for index in range(token_count - 1):
            if past is not None and index % block_size == 0:
                past = None
            probabilities, past = lib.next_probs(model, current, past)
            cumulative, total = lib.quantize(probabilities, vocabulary_size)
            encoder.encode_idx(cumulative, total, int(tokens[index + 1]))
            current = tokens[index + 1].view(1, 1)
            if (index + 1) % 2_000 == 0:
                speed = (index + 1) / max(time.time() - started_at, 1e-9)
                print(
                    f"  encode {index + 1}/{token_count - 1} ({speed:.0f} tok/s)",
                    flush=True,
                )
    return encoder.finish(), token_count


def encode_batched(
    text: str,
    model_id: str,
    scale_bits: int,
    revision: str | None = None,
) -> tuple[bytes, int]:
    """Encode with teacher forcing for experimental drift measurements.

    This path is not portable to the step-by-step decoder on all floating-point
    profiles. Every result must be verified by a full decode and CRC check.
    """
    tokenizer, model, lib = get_codec(model_id, revision)
    vocabulary_size = model.config.vocab_size
    _validate_scale(scale_bits, vocabulary_size, lib.PRECISION)
    lib.SCALE = 1 << scale_bits
    block_size = model.config.n_positions
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    token_count = int(tokens.shape[0])
    if token_count == 0:
        return b"", 0

    encoder = lib.FastEncoder()
    uniform_cumulative, uniform_total = lib.quantize(
        np.full(vocabulary_size, 1.0 / vocabulary_size), vocabulary_size
    )
    encoder.encode_idx(uniform_cumulative, uniform_total, int(tokens[0]))
    started_at = time.time()
    with torch.no_grad():
        for block_start in range(0, token_count - 1, block_size):
            segment = (
                tokens[block_start : block_start + block_size]
                .to(lib.DEVICE)
                .unsqueeze(0)
            )
            logits = model(segment).logits[0].float()
            segment_length = logits.shape[0]
            rows = (
                segment_length
                if block_start + segment_length < token_count
                else segment_length - 1
            )
            probabilities = torch.softmax(logits[:rows], dim=-1).cpu().numpy()
            for offset in range(rows):
                cumulative, total = lib.quantize(probabilities[offset], vocabulary_size)
                encoder.encode_idx(
                    cumulative, total, int(tokens[block_start + offset + 1])
                )
            done = min(block_start + rows, token_count - 1)
            speed = done / max(time.time() - started_at, 1e-9)
            print(f"  encode {done}/{token_count - 1} ({speed:.0f} tok/s)", flush=True)
    return encoder.finish(), token_count


def decode(
    payload: bytes,
    token_count: int,
    model_id: str,
    scale_bits: int,
    revision: str | None = None,
) -> str:
    """Decode an arithmetic-coded payload into text."""
    if token_count < 0:
        raise ContainerFormatError("token count cannot be negative")
    if token_count == 0:
        if payload:
            raise ContainerFormatError("empty streams cannot contain a payload")
        return ""

    tokenizer, model, lib = get_codec(model_id, revision)
    vocabulary_size = model.config.vocab_size
    _validate_scale(scale_bits, vocabulary_size, lib.PRECISION)
    lib.SCALE = 1 << scale_bits
    block_size = model.config.n_positions
    decoder = lib.FastDecoder(payload)
    uniform_cumulative, uniform_total = lib.quantize(
        np.full(vocabulary_size, 1.0 / vocabulary_size), vocabulary_size
    )
    output = [decoder.decode_idx(uniform_cumulative, uniform_total)]
    past = None
    current = torch.tensor([[output[0]]])
    started_at = time.time()
    with torch.no_grad():
        for index in range(token_count - 1):
            if past is not None and index % block_size == 0:
                past = None
            probabilities, past = lib.next_probs(model, current, past)
            cumulative, total = lib.quantize(probabilities, vocabulary_size)
            output.append(decoder.decode_idx(cumulative, total))
            current = torch.tensor([[output[-1]]])
            if (index + 1) % 2_000 == 0:
                speed = (index + 1) / max(time.time() - started_at, 1e-9)
                print(
                    f"  decode {index + 1}/{token_count - 1} ({speed:.0f} tok/s)",
                    flush=True,
                )
    return tokenizer.decode(output, clean_up_tokenization_spaces=False)


def pack(
    payload: bytes,
    token_count: int,
    crc: int,
    model_id: str,
    scale_bits: int,
    block: int = 2_048,
    model_revision: str | None = None,
    software_profile: str | None = None,
) -> bytes:
    """Pack a payload into a provenance-aware LLZ3 container."""
    from . import codec_lib

    if not isinstance(payload, bytes):
        raise ContainerFormatError("payload must be bytes")
    if token_count < 0:
        raise ContainerFormatError("token count cannot be negative")
    if not 0 <= crc <= 0xFFFFFFFF:
        raise ContainerFormatError("CRC32 must fit in uint32")
    if not 0 < block <= 0xFFFFFFFF:
        raise ContainerFormatError("block size must fit in uint32")
    if not 0 < scale_bits <= 255:
        raise ContainerFormatError("scale_bits must fit in uint8")

    model_bytes = model_id.encode("utf-8")
    if not model_bytes or len(model_bytes) > _MAX_MODEL_ID_BYTES:
        raise ContainerFormatError("model id length is outside the LLZ3 limit")
    revision = model_revision or codec_lib.resolve_model_revision(model_id)
    revision_bytes = revision.encode("ascii") if revision else b""
    if len(revision_bytes) > _MAX_REVISION_BYTES:
        raise ContainerFormatError("model revision is too long")
    software_bytes = (software_profile or _software_profile()).encode("ascii")
    if len(software_bytes) > 255:
        raise ContainerFormatError("software profile is too long")
    dtype_flag, device_flag = _current_profile()
    fixed = _LLZ3_FIXED.pack(
        len(model_bytes),
        len(revision_bytes),
        len(software_bytes),
        scale_bits,
        dtype_flag,
        device_flag,
        block,
        crc,
        token_count,
        len(payload),
    )
    return MAGIC3 + fixed + model_bytes + revision_bytes + software_bytes + payload


def _decode_utf8(value: bytes, field: str) -> str:
    """Decode a required UTF-8 container field."""
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContainerFormatError(f"{field} is not valid UTF-8") from error


def _validate_container_profile(
    scale_bits: int,
    dtype_flag: int,
    device_flag: int,
    block: int,
) -> None:
    """Reject impossible compute-profile metadata before model loading."""
    if not 1 <= scale_bits <= 28:
        raise ContainerFormatError(f"unsupported scale_bits value: {scale_bits}")
    if dtype_flag not in (0, 1, 2):
        raise ContainerFormatError(f"unsupported dtype flag: {dtype_flag}")
    if device_flag not in (0, 1, 2):
        raise ContainerFormatError(f"unsupported device flag: {device_flag}")
    if block == 0:
        raise ContainerFormatError("block size cannot be zero")


def unpack(buffer: bytes) -> tuple[str, int, int, int, bytes, Profile]:
    """Parse LLZ1, LLZ2, or LLZ3 without loading a model."""
    if not isinstance(buffer, bytes):
        raise ContainerFormatError("container must be bytes")
    if len(buffer) < 4:
        raise ContainerFormatError("container is shorter than its magic")

    magic = buffer[:4]
    if magic == MAGIC3:
        minimum = 4 + _LLZ3_FIXED.size
        if len(buffer) < minimum:
            raise ContainerFormatError("truncated LLZ3 fixed header")
        (
            model_length,
            revision_length,
            software_length,
            scale_bits,
            dtype_flag,
            device_flag,
            block,
            crc,
            token_count,
            payload_size,
        ) = _LLZ3_FIXED.unpack_from(buffer, 4)
        header_end = minimum + model_length + revision_length + software_length
        if header_end > len(buffer):
            raise ContainerFormatError("truncated LLZ3 model metadata")
        model_start = minimum
        revision_start = model_start + model_length
        software_start = revision_start + revision_length
        model_id = _decode_utf8(buffer[model_start:revision_start], "model id")
        revision_raw = buffer[revision_start:software_start]
        try:
            revision = revision_raw.decode("ascii") if revision_raw else None
        except UnicodeDecodeError as error:
            raise ContainerFormatError("model revision is not ASCII") from error
        software_raw = buffer[software_start:header_end]
        try:
            software = software_raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise ContainerFormatError("software profile is not ASCII") from error
        payload = buffer[header_end:]
        if len(payload) != payload_size:
            raise ContainerFormatError(
                f"payload size mismatch: header={payload_size}, actual={len(payload)}"
            )
        if not model_id:
            raise ContainerFormatError("model id cannot be empty")
        _validate_container_profile(scale_bits, dtype_flag, device_flag, block)
        return (
            model_id,
            scale_bits,
            crc,
            token_count,
            payload,
            {
                "dtype": dtype_flag,
                "device": device_flag,
                "block": block,
                "model_revision": revision,
                "container_version": 3,
                "payload_size": payload_size,
                "software_profile": software,
            },
        )

    if magic == MAGIC2:
        if len(buffer) < 5:
            raise ContainerFormatError("truncated LLZ2 header")
        model_length = buffer[4]
        model_end = 5 + model_length
        tail_end = model_end + _LLZ2_PROFILE.size + _LLZ2_TAIL.size
        if len(buffer) < tail_end:
            raise ContainerFormatError("truncated LLZ2 header")
        model_id = _decode_utf8(buffer[5:model_end], "model id")
        if not model_id:
            raise ContainerFormatError("model id cannot be empty")
        scale_bits, dtype_flag, device_flag, block = _LLZ2_PROFILE.unpack_from(
            buffer, model_end
        )
        crc, token_count = _LLZ2_TAIL.unpack_from(
            buffer, model_end + _LLZ2_PROFILE.size
        )
        _validate_container_profile(scale_bits, dtype_flag, device_flag, block)
        payload = buffer[tail_end:]
        return (
            model_id,
            scale_bits,
            crc,
            token_count,
            payload,
            {
                "dtype": dtype_flag,
                "device": device_flag,
                "block": block,
                "model_revision": None,
                "container_version": 2,
                "payload_size": len(payload),
            },
        )

    if magic == MAGIC:
        if len(buffer) < 5:
            raise ContainerFormatError("truncated LLZ1 header")
        model_length = buffer[4]
        model_end = 5 + model_length
        tail_end = model_end + _LLZ1_TAIL.size
        if len(buffer) < tail_end:
            raise ContainerFormatError("truncated LLZ1 header")
        model_id = _decode_utf8(buffer[5:model_end], "model id")
        if not model_id:
            raise ContainerFormatError("model id cannot be empty")
        scale_bits, crc, token_count = _LLZ1_TAIL.unpack_from(buffer, model_end)
        _validate_container_profile(scale_bits, 0, 0, 2_048)
        payload = buffer[tail_end:]
        return (
            model_id,
            scale_bits,
            crc,
            token_count,
            payload,
            {
                "dtype": 0,
                "device": 0,
                "block": 2_048,
                "model_revision": None,
                "container_version": 1,
                "payload_size": len(payload),
            },
        )

    raise ContainerFormatError("not an LLZ1, LLZ2, or LLZ3 container")


def profile_mismatch(profile: Profile) -> str:
    """Describe differences between container and active compute profiles."""
    dtype_flag, device_flag = _current_profile()
    messages: list[str] = []
    if profile.get("dtype") != dtype_flag:
        messages.append(f"archive dtype={profile.get('dtype')} vs current={dtype_flag}")
    if profile.get("device") != device_flag:
        messages.append(
            f"archive device={profile.get('device')} vs current={device_flag}"
        )
    archive_software = profile.get("software_profile")
    if archive_software and archive_software != _software_profile():
        messages.append("software versions differ")
    return "; ".join(messages)


def _print_container(path: Path) -> None:
    """Print container metadata without loading its model."""
    model_id, scale, crc, tokens, payload, profile = unpack(path.read_bytes())
    print(f"container: LLZ{profile['container_version']}")
    print(f"model: {model_id}")
    print(f"model revision: {profile.get('model_revision') or 'not recorded'}")
    print(f"tokens: {tokens}")
    print(f"scale: 2^{scale}")
    print(f"payload: {len(payload)} bytes")
    print(f"crc32: {crc:08x}")
    print(f"software: {profile.get('software_profile') or 'not recorded'}")
    print(
        f"profile: dtype={profile.get('dtype')} device={profile.get('device')} "
        f"block={profile.get('block')}"
    )


def main() -> None:
    """Run the LLZ command-line interface."""
    parser = argparse.ArgumentParser(prog="llz")
    parser.add_argument("cmd", choices=["encode", "decode", "bench", "inspect"])
    parser.add_argument("src")
    parser.add_argument("-o", "--out")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--scale", type=int, default=28)
    parser.add_argument(
        "--batched",
        action="store_true",
        help="experimental teacher-forcing encoder; always verify its CRC",
    )
    parser.add_argument(
        "--allow-crc-mismatch",
        action="store_true",
        help="write decoded bytes even when CRC32 does not match",
    )
    arguments = parser.parse_args()
    source = Path(arguments.src)

    if arguments.cmd == "inspect":
        _print_container(source)
        return

    if arguments.cmd in ("encode", "bench"):
        raw = source.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SystemExit(f"input is not valid UTF-8: {error}") from error
        started_at = time.time()
        encoder = encode_batched if arguments.batched else encode
        payload, token_count = encoder(
            text, arguments.model, arguments.scale, arguments.revision
        )
        elapsed = time.time() - started_at
        crc = zlib.crc32(raw) & 0xFFFFFFFF
        ratio = len(raw) / max(len(payload), 1)
        print(
            f"\n{len(raw)} bytes -> {len(payload)} bytes "
            f"({ratio:.2f}:1), {token_count} tokens, {elapsed:.0f} s"
        )
        xz_payload = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
        bz2_payload = bz2.compress(raw, 9)
        print(
            f"  xz: {len(xz_payload)} ({len(raw) / max(len(xz_payload), 1):.2f}:1), "
            f"bz2: {len(bz2_payload)} ({len(raw) / max(len(bz2_payload), 1):.2f}:1)"
        )
        if arguments.cmd == "encode":
            output = (
                Path(arguments.out) if arguments.out else source.with_suffix(".llz")
            )
            container = pack(
                payload,
                token_count,
                crc,
                arguments.model,
                arguments.scale,
                block=get_codec(arguments.model, arguments.revision)[
                    1
                ].config.n_positions,
                model_revision=arguments.revision,
            )
            output.write_bytes(container)
            print(
                f"container: {len(container)} bytes "
                f"({len(raw) / max(len(container), 1):.2f}:1)"
            )
            print(f"written: {output}")
        return

    model_id, scale, crc, token_count, payload, profile = unpack(source.read_bytes())
    revision = profile.get("model_revision")
    print(
        f"container: LLZ{profile['container_version']}, model {model_id}, "
        f"{token_count} tokens, {len(payload)} payload bytes"
    )
    warning = profile_mismatch(profile)
    if warning:
        print(f"WARNING: compute profile mismatch ({warning}); CRC32 is authoritative")
    text = decode(payload, token_count, model_id, scale, revision)
    raw = text.encode("utf-8")
    actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
    if actual_crc != crc:
        print(f"CRC32 MISMATCH: expected {crc:08x}, decoded {actual_crc:08x}")
        if not arguments.allow_crc_mismatch:
            raise SystemExit(
                "refusing to write corrupt output; use --allow-crc-mismatch to override"
            )
    else:
        print("CRC32 OK - restored byte-for-byte")
    output = (
        Path(arguments.out) if arguments.out else source.with_suffix(".decoded.txt")
    )
    output.write_bytes(raw)
    print(f"written: {output}")


if __name__ == "__main__":
    main()
