"""Stage 4c: retrain the sentence normalizer (romanized Telugu -> Telugu script) on ALL corpora, and benchmark it
against the Stage 4b model on the same test sets.

    python stages/s4c_normalizer_allcorpus.py prepare
    python stages/s4c_normalizer_allcorpus.py train [--steps 12000] [--quick]
    python stages/s4c_normalizer_allcorpus.py bench --old <stage4b .pt> --new data/normalizer/sentence_normalizer_allcorpus.pt

Same model size and recipe as Stage 4b (4 layers, d=256, 12,000 steps of 64), so the only change is the DATA.
Training sources (every Telugu-script line becomes (romanized with our romanizer, Telugu) pairs, as in Stage 4b):
  movie_train   movie subtitles, training movies only (Stage 4b's data)
  dak_dev       Dakshina dev sentences (real human romanization)
  repaired      Sonnet + Gemini repaired scenes (training movies only)
  director      Gemini director-style scenes (minus 300 held-out scenes used as a test)
  everyday      Gemini everyday chats written in Telugu script (Roman-only chats have no Telugu target, so unused)
  wiki          Dakshina Telugu Wikipedia (any sentence in a Dakshina test/dev romanized set removed)
  alpaca        Telugu Alpaca outputs (local use only; non-commercial licence)
Each batch draws from the sources with fixed weights, so the big formal sources can't drown the spoken ones.

Test sets (never trained on):
  dak300        the 300 Dakshina sentences of the Tier 1 reading test (real human romanization, formal)
  dak_test      all Dakshina test sentences (same style)
  movie_test    turns from the unseen test movies, noisy romanization by our romanizer (spoken)
  director_real held-out director scenes, romanized by Gemini (a different writer than our romanizer; spoken)
"""
import glob
import gzip
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT, load_scenes, split_of  # noqa: E402

N = ROOT / "data" / "normalizer"
X = Path(os.environ.get("TLM_EXTRA", ROOT.parent))  # folder holding the Gemini/Claude corpora
DAK = Path(os.environ["DAKSHINA_TE"]).expanduser()
MODE = sys.argv[1]
TE = re.compile(r"[ఀ-౿]")
WEIGHTS = {"movie_train": 0.25, "repaired": 0.08, "director": 0.12, "everyday": 0.10, "dak_dev": 0.10, "wiki": 0.25, "alpaca": 0.10}


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?।])\s+|\n", text) if TE.search(s or "")]


if MODE == "prepare":
    from tlm.romanizer import LOANWORDS, romanize_word  # noqa: E402
    TOKEN = re.compile(r"[ఀ-౿]+|[^ఀ-౿]+")

    def pair(line, rng):  # identical to Stage 4b (stages/s4b_prepare.py)
        src, tgt = [], []
        for t in TOKEN.findall(line):
            if "ఀ" <= t[0] <= "౿":
                if t in LOANWORDS:
                    src.append("@")
                    tgt.append("@")
                else:
                    src.append(romanize_word(t, rng, False, rng.choice([0.0, 0.0, 0.5, 1.5])))
                    tgt.append(t)
            else:
                src.append(t)
                tgt.append(t)
        return "".join(src).lower(), "".join(tgt)

    def ok(s, t):
        return 3 <= len(s) <= 160 and 1 <= len(t) <= 160 and re.search(r"[a-z]", s)

    split = split_of()
    rows = {k: [] for k in WEIGHTS}
    # movie_train + dak_dev: Stage 4b's training file (movie train turns x3 + Dakshina dev x3)
    nat = open(DAK / "romanized/te.romanized.rejoined.dev.native.txt", encoding="utf-8").read().splitlines()
    rom = open(DAK / "romanized/te.romanized.rejoined.dev.roman.txt", encoding="utf-8").read().splitlines()
    devset = {(r.lower(), n) for r, n in zip(rom, nat)}
    for l in open(N / "sent_train.tsv", encoding="utf-8"):
        p = tuple(l.rstrip("\n").split("\t"))
        rows["dak_dev" if p in devset else "movie_train"].append(p)
    # Dakshina sentences to exclude from Wikipedia (test + dev romanized sets)
    banned = set()
    for sp in ("test", "dev"):
        banned |= set(open(DAK / "romanized" / f"te.romanized.rejoined.{sp}.native.txt", encoding="utf-8").read().splitlines())
    rng = random.Random(0)

    def add(src, line):
        for s in sentences(line):
            p = pair(s, random.Random(f"{src}:{s}"))
            if ok(*p):
                rows[src].append(p)

    for f in glob.glob(str(X / "claude_data_tasks/A_repair/out_batch_*.json")) + glob.glob(str(X / "gemini_data_tasks/A_repair/out_batch_*.json")):
        for sc in json.load(open(f)):
            if split.get(sc["scene_id"].rsplit("_SC_", 1)[0]) != "train" or not sc.get("usable"):
                continue
            for t in sc["turns"]:
                add("repaired", t.get("spoken") or t["text"])
    D = []
    for f in sorted(glob.glob(str(X / "synthetic_director_corpus/batches/*.json"))):
        d = json.load(open(f))
        D += d if isinstance(d, list) else d.get("scenes", [])
    random.Random(5).shuffle(D)
    held, rest = D[:300], D[300:]
    for sc in rest:
        for t in sc["turns"]:
            add("director", t["spoken"])
    real = []
    for sc in held:
        for t in sc["turns"]:
            r, g = t.get("tenglish", ""), t["spoken"]
            if r and len(r.split()) == len(g.split()):
                real.append((r.lower(), g))
    for f in glob.glob(str(X / "everyday_chat_corpus/outputs/*.json")):
        for sc in json.load(open(f)):
            if sc.get("script") == "telugu":
                for t in sc["turns"]:
                    add("everyday", t["text"])
    with gzip.open(X / "external_datasets/dakshina/dakshina_dataset_v1.0/te/native_script_wikipedia" / "te.wiki-filt.train.text.shuf.txt.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and line not in banned:
                add("wiki", line)
    for l in open(X / "external_datasets/telugu_alpaca/telugu_alpaca_clean.jsonl", encoding="utf-8"):
        add("alpaca", json.loads(l).get("te_output", ""))
    for k, v in rows.items():
        with open(N / f"all_{k}.tsv", "w", encoding="utf-8") as f:
            f.writelines(f"{s}\t{t}\n" for s, t in v)
        print(f"{k}: {len(v):,} sentences, {sum(len(t.split()) for _, t in v):,} words")
    with open(N / "sent_director_real_test.tsv", "w", encoding="utf-8") as f:
        f.writelines(f"{s}\t{t}\n" for s, t in real)
    print("director_real test:", len(real))
    sys.exit(0)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tlm.seq2seq import PAD, Seq2Seq, Vocab  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
read = lambda f: [tuple(l.rstrip("\n").split("\t"))[:2] for l in open(N / f, encoding="utf-8")]  # noqa: E731
ML = 162


def pad(xs):
    return torch.tensor([x + [PAD] * (max(map(len, xs)) - len(x)) for x in xs], device=dev)


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    sv, tv = Vocab([]), Vocab([])
    sv.chars, tv.chars = ck["src"], ck["tgt"]
    sv.idx, tv.idx = {c: i + 3 for i, c in enumerate(sv.chars)}, {c: i + 3 for i, c in enumerate(tv.chars)}
    m = Seq2Seq(len(sv), len(tv), d=ck["d"], layers=ck["layers"], maxlen=ck["maxlen"]).to(dev)
    m.load_state_dict(ck["model"])
    return m.eval(), sv, tv, ck["maxlen"]


if MODE == "train":
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 12000
    quick = "--quick" in sys.argv
    if quick:
        steps = 200
    data = {k: read(f"all_{k}.tsv") for k in WEIGHTS}
    data = {k: v for k, v in data.items() if v}
    tests = [p for f in ("sent_movie_test.tsv", "sent_dak_test.tsv", "sent_director_real_test.tsv") for p in read(f)]
    src_v = Vocab([s for v in data.values() for s, _ in v] + [s for s, _ in tests])
    tgt_v = Vocab([t for v in data.values() for _, t in v] + [t for _, t in tests])
    model = Seq2Seq(len(src_v), len(tgt_v), d=256, layers=4, maxlen=ML).to(dev)
    print("params", sum(p.numel() for p in model.parameters()), "device", dev, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-4, total_steps=steps, pct_start=0.05)
    keys = list(data)
    w = [WEIGHTS[k] for k in keys]
    rng, t0 = random.Random(0), time.time()
    for step in range(1, steps + 1):
        b = [rng.choice(data[k]) for k in rng.choices(keys, w, k=64)]
        src, tgt = pad([src_v.encode(s, ML) for s, _ in b]), pad([tgt_v.encode(t, ML) for _, t in b])
        logits = model(src, tgt[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=PAD, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps or (quick and step % 50 == 0):
            el = time.time() - t0
            print(f"step {step}/{steps} loss {loss.item():.3f}  {el / 60:.1f} min, eta {el / step * (steps - step) / 60:.0f} min", flush=True)
    torch.save({"model": model.state_dict(), "src": src_v.chars, "tgt": tgt_v.chars, "d": 256, "layers": 4, "maxlen": ML},
               N / ("sentence_normalizer_allcorpus_quick.pt" if quick else "sentence_normalizer_allcorpus.pt"))
    print("saved", flush=True)

if MODE == "bench":
    old = sys.argv[sys.argv.index("--old") + 1]
    new = sys.argv[sys.argv.index("--new") + 1]
    dak300 = [tuple(x) for x in json.load(open(ROOT / "data/realworld/frontier/dakshina_300.json", encoding="utf-8"))]
    sets = {"dak300 (Tier 1 reading test)": dak300, "dak_test (all)": read("sent_dak_test.tsv"),
            "movie_test (spoken, unseen movies)": read("sent_movie_test.tsv")[:2000],
            "director_real (Gemini's own romanization)": read("sent_director_real_test.tsv")[:2000]}

    def run(m, sv, tv, ml, pairs):
        out = []
        for i in range(0, len(pairs), 64):
            out += [tv.decode(y.tolist()) for y in m.greedy(pad([sv.encode(s[:ml - 2], ml) for s, _ in pairs[i:i + 64]]), maxlen=ml)]
        return out

    def wacc(pairs, preds):
        hit = tot = 0
        for (_, g), p in zip(pairs, preds):
            pw, gw = re.findall(r"[ఀ-౿]+", p), re.findall(r"[ఀ-౿]+", g)
            hit += sum(x == y for x, y in zip(pw, gw))
            tot += len(gw)
        return round(hit / max(tot, 1), 4)

    res, ex = {}, {}
    for name, path in (("old (Stage 4b)", old), ("new (all corpora)", new)):
        m, sv, tv, ml = load_model(path)
        with torch.no_grad():
            for sname, pairs in sets.items():
                preds = run(m, sv, tv, ml, pairs)
                res.setdefault(sname, {})[name] = wacc(pairs, preds)
                ex.setdefault(sname, {})[name] = preds[:15]
                print(sname, name, res[sname][name], flush=True)
    json.dump(res, open(ROOT / "docs/results/s4c_normalizer_allcorpus.json", "w"), indent=1)
    json.dump({"inputs": {k: v[:15] for k, v in sets.items()}, "preds": ex}, open(N / "s4c_examples.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, indent=1))
