"""Fetch the Russian Wikipedia articles used by the LLZ benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API_URL = "https://ru.wikipedia.org/w/api.php"
USER_AGENT = "llz-codec/0.2 reproducibility benchmark"
RELEASE_SHA256 = "87c32bcab75083d2191c6592218748c7cff492e98cf5ce336ea7669088c6a5dc"
TITLES = [
    "История России",
    "Вторая мировая война",
    "Москва",
    "Русский язык",
    "Сжатие данных",
    "Эволюция",
    "Солнечная система",
    "Искусственный интеллект",
]


def fetch_article(title: str) -> str:
    """Fetch one current plain-text Wikipedia extract."""
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
        }
    )
    request = urllib.request.Request(  # noqa: S310 - API_URL is a fixed HTTPS origin
        f"{API_URL}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        payload = json.load(response)
    pages = payload.get("query", {}).get("pages", [])
    if not pages or "extract" not in pages[0]:
        raise RuntimeError(f"Wikipedia returned no extract for {title!r}")
    return str(pages[0]["extract"])


def main() -> None:
    """Fetch all articles and report whether they match the release corpus."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("corpus-current.txt"))
    arguments = parser.parse_args()

    sections: list[str] = []
    for title in TITLES:
        extract = fetch_article(title)
        sections.append(f"\n\n=== {title} ===\n\n{extract}")
        print(f"{title}: {len(extract)} characters", flush=True)
        time.sleep(0.3)

    raw = "".join(sections).encode("utf-8")
    arguments.output.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    print(f"\nTotal: {len(raw)} bytes; SHA-256 {digest}")
    if digest == RELEASE_SHA256:
        print("Matches the release benchmark corpus byte-for-byte.")
    else:
        print(
            "Does not match the release corpus because Wikipedia changed. "
            "Use benchmarks/corpus.txt for exact reproduction."
        )


if __name__ == "__main__":
    main()
