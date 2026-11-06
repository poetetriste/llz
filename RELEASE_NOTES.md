# llz v0.1.0 — релизная заметка / Release notes

## Русский

### Что это

`llz` — lossless-компрессор текста: языковая модель предсказывает следующий
токен, арифметический кодер записывает только «удивление». Каждый результат
подтверждён полным кругом encode→decode с побайтовой сверкой и CRC32.

### Главные цифры (все воспроизводимы, см. README.md §3)

| Утверждение | Значение | Проверка |
|---|---|---|
| Русская вики, 1 251 867 Б | **16,13 : 1** (77 590 Б) | CRC32 OK, полный круг |
| Лучшая классика на том же файле (bz2 -9) | 5,88 : 1 | — |
| Кросс-машинный lossless (два разных ПК) | OK на 162 и 2 968 токенах | CRC32 OK |
| Скорость (rugpt3large, RTX 3060, fp16+compile) | ~100 ток/с | замерено |

### Что нового относительно опубликованных работ (LLMZip, FineZip)

1. **CRC32-верификация каждого архива** — они показывали только таблицы.
2. **Кросс-машинная воспроизводимость** — измерено, что fp-инференс
   недетерминирован между машинами (fp32, fp64, пиннинг версий — всё
   расходилось), и найден рабочий рецепт: fp64 + MKL_CBWR=COMPATIBLE +
   ATEN_CPU_CAPABILITY=avx2 + фиксированные потоки + scale 2^18 +
   пиннинг версий torch/transformers/numpy. Цена: +2,8% размера.
3. **Честные отрицательные результаты** (см. REPORT.md): LoRA-под-документ
   не окупается < 10 МБ; мультиязычные модели слабее для русского; батчевый
   encode ломает арифметический кодер fp-дрейфом; доменное дообучение на
   28 КБ даёт +9%, а не +42%.

### Известные ограничения

- Степень зависит от домена: вики/новости 16–18×, юридические 11×,
  переписка 5,7×, код 4,9×.
- Скорость ~100 ток/с — холодное хранение и узкие каналы, не потоки.
- Без той же модели архив не читается.

---

## English

### What it is

`llz` is a lossless text compressor: a language model predicts the next token,
an arithmetic coder records only the "surprise". Every reported number is
verified by a full encode→decode round trip with byte-exact comparison (CRC32).

### Headline numbers (all reproducible, see README.md §3)

| Claim | Value | Verification |
|---|---|---|
| Russian Wikipedia corpus, 1,251,867 B | **16.13 : 1** (77,590 B) | CRC32 OK |
| Best classic on the same file (bz2 -9) | 5.88 : 1 | — |
| Cross-machine lossless (two different PCs) | OK at 162 and 2,968 tokens | CRC32 OK |
| Speed (rugpt3large, RTX 3060, fp16+compile) | ~100 tok/s | measured |

### What is new vs published work (LLMZip, FineZip)

1. **CRC32 verification of every archive** — they showed tables only.
2. **Cross-machine reproducibility** — we measured that fp inference is
   non-deterministic across machines (fp32, fp64, version pinning all failed)
   and found a working recipe: fp64 + MKL_CBWR=COMPATIBLE +
   ATEN_CPU_CAPABILITY=avx2 + pinned thread count + scale 2^18 +
   pinned torch/transformers/numpy versions. Cost: +2.8% size.
3. **Honest negative results** (see REPORT.md): per-document LoRA does not
   pay off below 10 MB; multilingual models underperform for Russian; batched
   teacher-forcing encoding breaks the arithmetic coder via fp drift; domain
   fine-tuning on 28 KB yields +9%, not +42%.

### Known limitations

- Ratio is domain-dependent: wiki/news 16–18×, legal 11×, chat 5.7×, code 4.9×.
- ~100 tok/s — cold storage and narrow channels, not streaming.
- An archive is unreadable without the exact same model.
