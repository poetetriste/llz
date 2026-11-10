# -*- coding: utf-8 -*-
"""
llz — lossless-сжатие текста языковой моделью + арифметическим кодированием.

Публичный API:
    from llz import encode, decode, pack, unpack
    blob, n = encode(text, model_id, scale_bits)
    payload = pack(blob, n, crc32, model_id, scale_bits)
    ...
CLI:  python -m llz.cli encode book.txt   (или entry point `llz`)
"""

from .cli import (
    DEFAULT_MODEL,
    MAGIC,
    MAGIC2,
    MAGIC3,
    decode,
    encode,
    encode_batched,
    get_codec,
    pack,
    profile_mismatch,
    unpack,
)

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_MODEL",
    "MAGIC",
    "MAGIC2",
    "MAGIC3",
    "decode",
    "encode",
    "encode_batched",
    "get_codec",
    "pack",
    "profile_mismatch",
    "unpack",
    "__version__",
]
