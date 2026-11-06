# -*- coding: utf-8 -*-
"""
finetune_domain.py — тест «доменной модели как словаря кодека».
Полное дообучение rugpt3small на train-части домена, замер энтропии
на held-out ДО и ПОСЛЕ.
"""
import math, os, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "sberbank-ai/rugpt3small_based_on_gpt2"
DOC = os.environ.get("FT_DOC", "dom_chat.txt")
EPOCHS = int(os.environ.get("FT_EPOCHS", "5"))
LR = float(os.environ.get("FT_LR", "2e-5"))
BLOCK = 512
OUT = os.environ.get("FT_OUT", "rugpt3small-chat")

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda()

ids = tok(open(DOC, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
n = ids.shape[0]
TRAIN_END = float(os.environ.get("FT_TRAIN_END", "0.6"))   # честный сплит
TEST_START = float(os.environ.get("FT_TEST_START", "0.8"))  # зазор 60-80% выброшен
tr, te = ids[:int(n*TRAIN_END)], ids[int(n*TEST_START):]
print(f"{DOC}: {n} токенов (train {tr.shape[0]}, test {te.shape[0]})", flush=True)

@torch.no_grad()
def bpt(model, tids):
    tot, cnt = 0.0, 0
    for b in range(tids.shape[0] // BLOCK):
        x = tids[b*BLOCK:(b+1)*BLOCK].unsqueeze(0).cuda()
        out = model(input_ids=x, labels=x)
        tot += out.loss.item() * (x.shape[1]-1); cnt += x.shape[1]-1
    return tot / cnt / math.log(2)

m.eval()
b0 = bpt(m, te)
print(f"ДО:  {b0:.3f} бит/токен на held-out", flush=True)

m.train()
opt = torch.optim.AdamW(m.parameters(), lr=LR)
nb = tr.shape[0] // BLOCK
t0 = time.time()
step = 0
for ep in range(EPOCHS):
    for b in range(nb):
        x = tr[b*BLOCK:(b+1)*BLOCK].unsqueeze(0).cuda()
        loss = m(input_ids=x, labels=x).loss
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
    print(f"  эпоха {ep+1}/{EPOCHS} loss={loss.item():.3f} ({step/(time.time()-t0):.1f} ст/с)", flush=True)

m.eval()
b1 = bpt(m, te)
print(f"ПОСЛЕ: {b1:.3f} бит/токен на held-out (выигрыш {100*(1-b1/b0):.1f}%)", flush=True)

BYTES_PER_TOK = 34729 / 11432
r0 = BYTES_PER_TOK * 8 / b0
r1 = BYTES_PER_TOK * 8 / b1
print(f"оценка степени на чате: было {r0:.2f}x -> стало {r1:.2f}x", flush=True)

if b1 < b0 * 0.9:
    m.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"модель сохранена: {OUT}/", flush=True)
else:
    print("выигрыш < 10% — модель не сохраняю", flush=True)
