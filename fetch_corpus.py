# -*- coding: utf-8 -*-
"""Качаем длинные русские статьи Википедии как честный корпус."""
import requests, time

TITLES = [
    "История России", "Вторая мировая война", "Москва",
    "Русский язык", "Сжатие данных", "Эволюция",
    "Солнечная система", "Искусственный интеллект",
]

corpus = []
for t in TITLES:
    r = requests.get(
        "https://ru.wikipedia.org/w/api.php",
        params={"action": "query", "prop": "extracts", "explaintext": 1,
                "titles": t, "format": "json", "redirects": 1},
        headers={"User-Agent": "compression-lab/0.1"}, timeout=30)
    pages = r.json()["query"]["pages"]
    text = next(iter(pages.values())).get("extract", "")
    corpus.append(f"\n\n=== {t} ===\n\n{text}")
    print(f"{t}: {len(text)} зн.", flush=True)
    time.sleep(0.3)

full = "".join(corpus)
with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write(full)
print(f"\nИТОГО: {len(full)} зн, {len(full.encode('utf-8'))} байт")
