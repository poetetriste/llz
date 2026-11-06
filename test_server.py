# -*- coding: utf-8 -*-
"""test_server.py — end-to-end тест HTTP-сервиса llz_server (in-process)."""
import json, threading, urllib.request, urllib.error, zlib
from http.server import ThreadingHTTPServer
from llz import server as llz_server

PORT = 18080
srv = ThreadingHTTPServer(("127.0.0.1", PORT), llz_server.Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()

BASE = f"http://127.0.0.1:{PORT}"
FAILS = []
def check(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""), flush=True)
    if not cond:
        FAILS.append(name)

def post(path, data: bytes):
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    try:
        return urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        return e

# 1. healthz
r = urllib.request.urlopen(BASE + "/healthz", timeout=10)
h = json.loads(r.read())
check("healthz", h["status"] == "ok" and h["model"] == llz_server.MODEL)

# 2. главная страница
r = urllib.request.urlopen(BASE + "/", timeout=10)
check("GET / отдаёт страницу", b"llz" in r.read())

# 3. encode -> decode круг
texts = [
    "Москва — столица России. " * 10,
    "Короткий тест 👋 с emoji и 数字.",
    "a",
    "",
]
for i, t in enumerate(texts):
    raw = t.encode("utf-8")
    r1 = post("/encode", raw)
    ok = r1.status == 200
    blob = r1.read()
    check(f"S{i}: encode HTTP 200", ok)
    if not ok:
        continue
    r2 = post("/decode", blob)
    back = r2.read()
    check(f"S{i}: decode CRC ok + побайтово",
          r2.status == 200 and r2.headers.get("X-CRC32") == "ok" and back == raw,
          f"{len(raw)} -> {len(blob)} байт")

# 4. битый UTF-8 -> 400
r = post("/encode", b"\xff\xfe\x00bad")
check("S: битый UTF-8 -> 400", r.status == 400)

# 5. мусор вместо контейнера -> 400
r = post("/decode", b"garbage-data-here")
check("S: мусор в /decode -> 400", r.status == 400)

# 6. повреждённый контейнер -> CRC MISMATCH (но не падение сервера)
t = "Проверка повреждения данных в контейнере."
r1 = post("/encode", t.encode("utf-8"))
blob = bytearray(r1.read())
blob[-5] ^= 0xFF
r2 = post("/decode", bytes(blob))
check("S: повреждённый контейнер -> MISMATCH, сервер жив",
      r2.status == 200 and r2.headers.get("X-CRC32") == "MISMATCH")

# 7. 404
try:
    urllib.request.urlopen(BASE + "/nope", timeout=5)
    check("S: 404 на неизвестный путь", False)
except urllib.error.HTTPError as e:
    check("S: 404 на неизвестный путь", e.code == 404)

srv.shutdown()
print("\n" + ("ВСЕ ТЕСТЫ ЗЕЛЁНЫЕ" if not FAILS else f"ПРОВАЛЫ: {FAILS}"))
exit(1 if FAILS else 0)
