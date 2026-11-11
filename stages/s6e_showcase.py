"""Stage 6e — showcase: 20 everyday romanized messages (10 types) answered by every model, side by side.

    python stages/s6e_showcase.py gpu --adapter ... --sentence-normalizer ...     # ours + untuned sarvam-1 (GPU)
    python stages/s6e_showcase.py cli                                             # Claude Haiku / Sonnet / Opus (CLI)
    python stages/s6e_showcase.py build [--gemini path/to/gemini_5_showcase.json] # writes docs/eval/SHOWCASE.md

Every model gets the message exactly as typed (romanized, mixed with English). Ours runs its own pipeline (English detection ->
sentence normalizer -> fine-tuned reply model); untuned sarvam-1 gets the same normalized text. Frontier models are asked
to (1) write the message in Telugu script and (2) reply in spoken Telugu, in one JSON answer.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from showcase_prompts import SHOWCASE  # noqa: E402
from tlm.corpus import ROOT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("phase", choices=["gpu", "cli", "build"])
ap.add_argument("--adapter")
ap.add_argument("--sentence-normalizer")
ap.add_argument("--gemini")
a = ap.parse_args()
OUT = ROOT / "data" / "realworld" / "showcase"
OUT.mkdir(parents=True, exist_ok=True)
msgs = [m for _, m in SHOWCASE]
FRONTIER = ("Below are {n} chat messages typed by Telugu speakers in English letters (romanized Telugu, often mixed with English "
            "words). For each, give: \"telugu\": the message written in Telugu script (keep English words as they are), and \"reply\": "
            "a short, natural reply in spoken, colloquial Telugu (Telugu script, under 20 words), as a friend or family member would "
            "really reply. Return ONLY a JSON list of {n} objects with keys \"telugu\" and \"reply\", in the same order.\n\n{items}")

if a.phase == "gpu":
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tlm.langid import LangID
    from tlm.seq2seq import Seq2Seq, Vocab

    ck = torch.load(a.sentence_normalizer, map_location="cuda")
    sv, tv = Vocab([]), Vocab([])
    sv.chars, tv.chars = ck["src"], ck["tgt"]
    sv.idx, tv.idx = {c: i + 3 for i, c in enumerate(sv.chars)}, {c: i + 3 for i, c in enumerate(tv.chars)}
    norm = Seq2Seq(len(sv), len(tv), d=ck["d"], layers=ck["layers"], maxlen=ck["maxlen"]).cuda()
    norm.load_state_dict(ck["model"])
    norm.eval()
    lid = LangID.load(ROOT / "data" / "normalizer" / "langid.json")
    EN = {"ok", "okay", "hi", "bro", "thanks", "sorry", "please", "plz", "yes", "no"}

    def understand(text):
        toks = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
        eng = [t for t in toks if t.isalpha() and (t.lower() in EN or lid.is_english(t))]
        masked = "".join("@" if (t.isalpha() and t in eng) else t.lower() for t in toks)
        o = tv.decode(norm.greedy(torch.tensor([sv.encode(masked, ck["maxlen"])], device="cuda"), maxlen=ck["maxlen"])[0].tolist())
        it = iter(eng)
        return re.sub("@", lambda g: next(it, "@"), o)

    tok = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-1", torch_dtype=torch.bfloat16,
                                                                           device_map="cuda"), a.adapter).eval()
    NL = tok("\n", add_special_tokens=False)["input_ids"][-1]
    BEST = dict(do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.2, no_repeat_ngram_size=3)

    @torch.no_grad()
    def reply(te, adapter, seed):
        torch.manual_seed(seed)
        x = tok(f"- {te}\n- ", return_tensors="pt").to("cuda")
        gen = lambda: model.generate(**x, max_new_tokens=50, eos_token_id=[tok.eos_token_id, NL],  # noqa: E731
                                     pad_token_id=tok.pad_token_id, **BEST)
        if adapter:
            y = gen()
        else:
            with model.disable_adapter():
                y = gen()
        return tok.decode(y[0][x["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0].strip()

    rows = []
    for i, m in enumerate(msgs):
        te = understand(m)
        rows.append({"ours": {"telugu": te, "reply": reply(te, True, i)},
                     "sarvam-1 untuned": {"telugu": te, "reply": reply(te, False, i)}})
    json.dump(rows, open(OUT / "gpu.json", "w"), ensure_ascii=False, indent=1)
    print("SHOWCASE_GPU_DONE")

elif a.phase == "cli":
    res = {}
    items = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(msgs))
    for name, mid in [("Claude Haiku 5.5", "claude-haiku-5-5"), ("Claude Sonnet 5.5", "claude-sonnet-5-5"),
                      ("Claude Opus 5.5", "claude-opus-5-5")]:
        for _ in range(3):
            out = subprocess.run(["claude", "-p", FRONTIER.format(n=len(msgs), items=items), "--model", mid,
                                  "--output-format", "text"], cwd=ROOT / "data" / "realworld" / "frontier" / "cli_workdir",
                                 capture_output=True, text=True, timeout=900).stdout
            m = re.search(r"\[.*\]", out, flags=re.S)
            try:
                lst = json.loads(m.group()) if m else []
            except json.JSONDecodeError:
                lst = []
            if len(lst) == len(msgs):
                res[name] = lst
                break
        print(name, "ok" if name in res else "failed", flush=True)
    json.dump(res, open(OUT / "claude.json", "w"), ensure_ascii=False, indent=1)
    task = {"instruction": FRONTIER.format(n=len(msgs), items="(see 'items')").replace("\n\n(see 'items')", "") +
            " Answer for the messages in 'items'.", "items": msgs}
    D = Path.home() / "Downloads" / "Tenglish_LM" / "frontier_tasks"
    json.dump(task, open(D / "5_showcase.json", "w"), ensure_ascii=False, indent=1)
    print("wrote", D / "5_showcase.json")

else:
    from tlm.romanizer import romanize

    gpu = json.load(open(OUT / "gpu.json", encoding="utf-8"))
    cl = json.load(open(OUT / "claude.json", encoding="utf-8"))
    gem = json.load(open(a.gemini, encoding="utf-8")) if a.gemini and Path(a.gemini).exists() else None
    models = ["ours", "sarvam-1 untuned"] + list(cl) + (["Gemini 3.8 Flash"] if gem else [])

    def get(i, m):
        if m in ("ours", "sarvam-1 untuned"):
            return gpu[i][m]
        if m == "Gemini 3.8 Flash":
            return gem[i]
        return cl[m][i]

    md = ["# Showcase: 20 everyday messages, every model side by side", "",
          "Messages written for this project and typed the way people chat: romanized Telugu, mixed with English. "
          "Each model shows **what it understood** (Telugu script) and **its reply** (Telugu, plus a romanized reading). "
          "See [PROMPTS.md](PROMPTS.md) for the exact instructions and [README.md](README.md) for all scores.", "",
          "| Model | Size |", "|---|---|", "| ours | 7.6M normalizer + 2.5B sarvam-1 with a 24M fine-tuned add-on |",
          "| sarvam-1 untuned | 2.5B, no fine-tuning (same normalized input as ours) |",
          "| Claude Haiku / Sonnet / Opus 5.5 | not disclosed (via Claude Code CLI) |",
          "| Gemini 3.8 Flash | not disclosed (via Antigravity) |", ""]
    cur = None
    for i, (kind, m) in enumerate(SHOWCASE):
        if kind != cur:
            md += [f"## {kind}", ""]
            cur = kind
        md += [f"**Typed:** `{m}`", "", "| Model | understood as | reply | reply (romanized) |", "|---|---|---|---|"]
        for mod in models:
            r = get(i, mod)
            rep = str(r.get("reply", "")).replace("|", "/")
            md.append(f"| {mod} | {str(r.get('telugu', '')).replace('|', '/')} | {rep} | `{romanize(rep, canonical=True)}` |")
        md.append("")
    E = ROOT / "docs" / "eval"
    E.mkdir(parents=True, exist_ok=True)
    (E / "SHOWCASE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    json.dump({"messages": SHOWCASE, "answers": {mod: [get(i, mod) for i in range(len(SHOWCASE))] for mod in models}},
              open(E / "showcase.json", "w"), ensure_ascii=False, indent=1)
    print("wrote docs/eval/SHOWCASE.md with", len(models), "models")
