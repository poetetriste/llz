# -*- coding: utf-8 -*-
"""
llz_server.py — HTTP-сервис кодека llz (stdlib only, без внешних зависимостей).

  python llz_server.py [--port 8080] [--model sberbank-ai/rugpt3small_based_on_gpt2]

Эндпоинты:
  POST /encode   тело = исходный текст (UTF-8, сырые байты)
                 -> 200 application/octet-stream: контейнер .llz
                 заголовки: X-Orig-Bytes, X-Comp-Bytes, X-Ratio, X-Tokens, X-CRC32
  POST /decode   тело = контейнер .llz
                 -> 200 text/plain; charset=utf-8: восстановленный текст
                 заголовок X-CRC32: ok | MISMATCH
  GET  /         страница с честными бенчмарками и живым тестом
  GET  /healthz  {"status":"ok","model":...}

Ограничения (честно): вход /encode должен быть валидным UTF-8 текстом.
Бинарные данные пока не поддерживаем (токенизатор текстовый).
"""

import argparse
import html
import json
import os
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import cli as llz
from .errors import ContainerFormatError

MODEL = llz.DEFAULT_MODEL
REVISION = None
MAX_BODY_BYTES = int(os.environ.get("LLZ_MAX_BODY_BYTES", str(16 * 1024 * 1024)))
MAX_DECODE_TOKENS = int(os.environ.get("LLZ_MAX_DECODE_TOKENS", "2000000"))
_lock = threading.Lock()  # одна модель — один инференс за раз

BENCH_ROWS = [
    # (корпус, размер, llz, xz/gzip, bz2, примечание) — только проверенные кругом цифры
    (
        "Русская Википедия",
        "1 251 867 Б",
        "77 590 Б payload (16,13×)",
        "xz 5,21×",
        "bz2 5,88×",
        "полный круг, CRC32 OK",
    ),
    (
        "Русская Википедия",
        "110 КБ",
        "6 134 Б (17,93×)",
        "xz 4,39×",
        "bz2 5,20×",
        "rugpt3large; полный круг",
    ),
    (
        "Письмо (рус.)",
        "1 109 Б",
        "141 Б payload (7,87×)",
        "—",
        "—",
        "rugpt3small; полный круг",
    ),
    (
        "Самосгенерированный зацикленный текст",
        "39 277 Б",
        "147 Б payload (267,19×)",
        "xz 39,59×",
        "bz2 40,87×",
        "полный круг; энтропийная оценка 341,26×",
    ),
]

PAGE = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>llz — LLM-компрессор</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;background:#0f1115;color:#e8e8e8}
h1{font-size:1.5em} h2{font-size:1.1em;color:#9ecfff}
table{border-collapse:collapse;width:100%;margin:1em 0}
td,th{border:1px solid #333;padding:6px 10px;text-align:left;font-size:.9em}
th{background:#1a1d24}
textarea{width:100%;height:120px;background:#16181d;color:#e8e8e8;border:1px solid #333;padding:8px}
button{background:#2b6cb0;color:#fff;border:0;padding:8px 18px;cursor:pointer;border-radius:4px}
pre{background:#16181d;padding:10px;overflow:auto;font-size:.85em}
.note{color:#888;font-size:.85em}
</style></head><body>
<h1>llz — lossless-сжатие текста языковой моделью</h1>
<p>Арифметическое кодирование поверх распределений LLM. Восстановление байт-в-байт (CRC32 в контейнере).
Модель: <b>__MODEL__</b>. Декодеру нужна та же модель.</p>

<h2>Живой тест</h2>
<textarea id="t">Москва — столица России. Проверяем сжатие вживую через HTTP API.</textarea><br><br>
<button onclick="go()">Сжать и восстановить</button>
<pre id="out">—</pre>

<h2>Замеренные результаты (без приукрашивания)</h2>
<table><tr><th>Корпус</th><th>Размер</th><th>llz</th><th>xz/gzip</th><th>brotli/bz2</th><th>Примечание</th></tr>
__ROWS__
</table>
<p class="note">Коэффициенты — только поток сжатия, без учёта весов модели (модель ~500 МБ у rugpt3small;
считается предустановленной у обеих сторон). Числа воспроизводимы: <code>python test_llz.py A</code>,
<code>python llz.py bench &lt;файл&gt;</code>.</p>

<script>
async function go(){
  const t = document.getElementById('t').value;
  const r1 = await fetch('/encode', {method:'POST', body:new TextEncoder().encode(t)});
  if(!r1.ok){ out.textContent = 'encode error: ' + await r1.text(); return; }
  const blob = await r1.arrayBuffer();
  const r2 = await fetch('/decode', {method:'POST', body:blob});
  const back = await r2.text();
  out.textContent =
    `исходно: ${r1.headers.get('X-Orig-Bytes')} байт\\n` +
    `сжато:   ${r1.headers.get('X-Comp-Bytes')} байт (${r1.headers.get('X-Ratio')} : 1)\\n` +
    `токенов: ${r1.headers.get('X-Tokens')}\\n` +
    `decode CRC32: ${r2.headers.get('X-CRC32')}\\n` +
    `совпадение: ${back === t ? 'ДА, байт-в-байт' : 'НЕТ!'}\\n` +
    `---\\n${back}`;
}
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/plain; charset=utf-8", headers=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            self._send(
                200,
                json.dumps({"status": "ok", "model": MODEL, "revision": REVISION}),
                "application/json",
            )
        elif self.path == "/":
            rows = "\n".join(
                f"<tr><td>{html.escape(corpus)}</td><td>{html.escape(size)}</td>"
                f"<td>{html.escape(llz_result)}</td><td>{html.escape(classic)}</td>"
                f"<td>{html.escape(other)}</td><td>{html.escape(note)}</td></tr>"
                for corpus, size, llz_result, classic, other, note in BENCH_ROWS
            )
            self._send(
                200,
                PAGE.replace("__MODEL__", html.escape(MODEL)).replace("__ROWS__", rows),
                "text/html; charset=utf-8",
            )
        else:
            self._send(404, "not found")

    def do_POST(self):
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            self._send(411, "Content-Length required")
            return
        try:
            n = int(length_header)
        except ValueError:
            self._send(400, "invalid Content-Length")
            return
        if n < 0:
            self._send(400, "invalid Content-Length")
            return
        if n > MAX_BODY_BYTES:
            self._send(413, "request body too large")
            return
        raw = self.rfile.read(n)
        if len(raw) != n:
            self._send(400, "truncated request body")
            return
        if self.path == "/encode":
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                self._send(400, "вход должен быть валидным UTF-8 текстом")
                return
            t0 = time.time()
            with _lock:
                payload, ntok = llz.encode(text, MODEL, 28, REVISION)
                block = llz.get_codec(MODEL, REVISION)[1].config.n_positions
                blob = llz.pack(
                    payload,
                    ntok,
                    zlib.crc32(raw) & 0xFFFFFFFF,
                    MODEL,
                    28,
                    block=block,
                    model_revision=REVISION,
                )
            self._send(
                200,
                blob,
                "application/octet-stream",
                {
                    "X-Orig-Bytes": str(len(raw)),
                    "X-Comp-Bytes": str(len(blob)),
                    "X-Ratio": f"{len(raw) / max(1, len(blob)):.2f}",
                    "X-Tokens": str(ntok),
                    "X-CRC32": f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}",
                    "X-Encode-Sec": f"{time.time() - t0:.1f}",
                },
            )
        elif self.path == "/decode":
            try:
                mid, scale, crc, ntok, payload, prof = llz.unpack(raw)
            except ContainerFormatError:
                self._send(400, "malformed LLZ container")
                return
            if mid != MODEL:
                self._send(409, f"контейнер под модель {mid}, у сервера {MODEL}")
                return
            if ntok > MAX_DECODE_TOKENS:
                self._send(413, "archive token count exceeds server limit")
                return
            t0 = time.time()
            with _lock:
                text = llz.decode(payload, ntok, mid, scale, prof.get("model_revision"))
            back = text.encode("utf-8")
            ok = (zlib.crc32(back) & 0xFFFFFFFF) == crc
            if not ok:
                self._send(
                    422, "decoded CRC32 mismatch", headers={"X-CRC32": "MISMATCH"}
                )
                return
            self._send(
                200,
                back,
                "text/plain; charset=utf-8",
                {
                    "X-CRC32": "ok" if ok else "MISMATCH",
                    "X-Decode-Sec": f"{time.time() - t0:.1f}",
                },
            )
        else:
            self._send(404, "not found")

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}", flush=True)


def main():
    global MODEL, REVISION
    ap = argparse.ArgumentParser(prog="llz-server")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", default=llz.DEFAULT_MODEL)
    ap.add_argument("--revision")
    a = ap.parse_args()
    MODEL = a.model
    REVISION = a.revision
    print(f"загружаю модель {MODEL} ...", flush=True)
    llz.get_codec(MODEL, REVISION)  # прогрев
    print(f"слушаю http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
