# LLZ repository guidance

- Arithmetic-coded archives depend on the exact integer probability trace.
  Do not normalize, reorder, or otherwise alter valid model probabilities as a
  cleanup; even tiny changes break backward decoding.
- Every compression claim must distinguish entropy estimate, arithmetic payload,
  and full container size.
- A measured lossless claim requires encode → decode, byte comparison, and CRC32.
- Preserve LLZ1/LLZ2 parsing when changing LLZ3.
- Run before release:
  - `python -m unittest discover -s tests -v`
  - `python -O test_llz.py A`
  - `python test_llz.py B 0 99`
  - `python test_server.py`
  - portable `cross_test/decode_test.py letter.cpu.llz`
- `verify_big.py` checkpoints are trusted local pickle files. Never load a
  checkpoint from an untrusted source.
- Benchmark corpora and archives are identified by SHA-256 in
  `benchmarks/manifest.json`; changing them requires new results and hashes.
