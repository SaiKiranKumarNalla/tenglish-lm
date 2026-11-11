"""Stage 6 — real-world evaluation: real romanized messages + replies judged by a stronger model.

    python stages/s6_real_world_eval.py --phase generate --adapter ... --sentence-normalizer ...   # sarvam-1, ~20 min
    python stages/s6_real_world_eval.py --phase judge                                              # Sarvam-M judge

Phase "generate"
  * real YouTube comments (romanized, typed by people) go through the pipeline (English detection -> sentence
    normalizer). With no gold Telugu for comments, understanding is checked by ROUND-TRIP: Telugu output ->
    canonical romanization -> phonetic key, compared word by word with the phonetic key of what was typed.
    The same check on Dakshina sentences (where the true answer is known) calibrates what the number means.
  * replies to the real comments from sarvam-1 as-is and from the fine-tuned model (best decoding from Stage 5b)
  * judge-reliability cases from the unseen movies: real reply vs a random reply from another scene, etc.
Phase "judge"
  * Sarvam-M (open 24B Indic model) compares two replies to the same message and picks the more natural, relevant,
    spoken-Telugu reply. Every pair is judged in BOTH orders; a win counts only when both orders agree, otherwise tie.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.corpus import ROOT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--phase", choices=["generate", "judge"], required=True)
ap.add_argument("--base", default="sarvamai/sarvam-1")
ap.add_argument("--adapter")
ap.add_argument("--sentence-normalizer")
ap.add_argument("--judge", default="sarvamai/sarvam-m")
ap.add_argument("--n-comments", type=int, default=250)
ap.add_argument("--n-movie", type=int, default=150)
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
if a.quick:
    a.n_comments, a.n_movie = 8, 6
GEN = ROOT / "data" / "realworld" / "generated.json"
OUT = ROOT / "docs" / "results" / "s6_real_world.json"

if a.phase == "generate":
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tlm.langid import LangID
    from tlm.romanizer import romanize
    from tlm.seq2seq import Seq2Seq, Vocab
    from tlm.textnorm import _key

    dev = "cuda"
    rng = random.Random(0)
    comments = [json.loads(l)["text"] for l in open(ROOT / "data" / "realworld" / "comments.jsonl", encoding="utf-8")]
    comments = rng.sample(comments, min(a.n_comments, len(comments)))
    dak = [l.rstrip("\n").split("\t") for l in open(ROOT / "data" / "normalizer" / "sent_dak_test.tsv", encoding="utf-8")]
    dak = rng.sample(dak, min(300, len(dak)))
    test = [json.loads(l) for l in open(ROOT / "data" / "dialogue" / "test.jsonl", encoding="utf-8")]

    # --- normalizer pipeline ---
    ck = torch.load(a.sentence_normalizer, map_location=dev)
    sv, tv = Vocab([]), Vocab([])
    sv.chars, tv.chars = ck["src"], ck["tgt"]
    sv.idx, tv.idx = {c: i + 3 for i, c in enumerate(sv.chars)}, {c: i + 3 for i, c in enumerate(tv.chars)}
    norm = Seq2Seq(len(sv), len(tv), d=ck["d"], layers=ck["layers"], maxlen=ck["maxlen"]).to(dev)
    norm.load_state_dict(ck["model"])
    norm.eval()
    lid = LangID.load(ROOT / "data" / "normalizer" / "langid.json")
    EN_CHAT = {"ok", "okay", "hi", "bro", "thanks", "sorry", "please", "plz", "yes", "no"}

    def understand(texts):
        masked, engs = [], []
        for text in texts:
            toks = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
            eng = [t for t in toks if t.isalpha() and (t.lower() in EN_CHAT or lid.is_english(t))]
            masked.append("".join("@" if (t.isalpha() and t in eng) else t.lower() for t in toks)[: ck["maxlen"] - 2])
            engs.append(eng)
        out = []
        for i in range(0, len(masked), 64):
            enc = [sv.encode(m, ck["maxlen"]) for m in masked[i:i + 64]]
            L = max(map(len, enc))
            src = torch.tensor([e + [0] * (L - len(e)) for e in enc], device=dev)
            out += [tv.decode(y.tolist()) for y in norm.greedy(src, maxlen=ck["maxlen"])]
        res = []
        for o, m, eng in zip(out, masked, engs):
            it = iter(eng)
            res.append((m, re.sub("@", lambda g: next(it, "@"), o)))
        return res

    def round_trip(masked, telugu):
        """Share of typed Telugu words whose phonetic key matches the key of the understood word (romanized back)."""
        typed = [w for w in re.findall(r"[a-z]+|@", masked)]
        back = [romanize(w, canonical=True).lower() for w in re.findall(r"[ఀ-౿]+|[A-Za-z]+", telugu)]
        if len(typed) != len(back):
            return None
        pairs = [(t, b) for t, b in zip(typed, back) if t != "@"]
        return sum(_key(t) == _key(b) for t, b in pairs) / max(len(pairs), 1)

    res = {}
    und_comments = understand(comments)
    und_dak = understand([r for r, _ in dak])
    for name, und, extra in [("real YouTube comments", und_comments, None), ("Dakshina sentences (gold known)", und_dak, dak)]:
        rts = [round_trip(m, t) for m, t in und]
        ok = [x for x in rts if x is not None]
        d = {"messages": len(und), "aligned": len(ok), "round_trip_word_match": round(sum(ok) / max(len(ok), 1), 4)}
        if extra:
            gold = [g for _, g in extra]
            hit = tot = 0
            for (_, t), g in zip(und, gold):
                tw, gw = re.findall(r"[ఀ-౿]+", t), re.findall(r"[ఀ-౿]+", g)
                hit += sum(x == y for x, y in zip(tw, gw))
                tot += len(gw)
            d["true_word_accuracy"] = round(hit / tot, 4)
        res[name] = d
    words = [w for c in comments for w in re.findall(r"[A-Za-z]+", c)]
    res["real YouTube comments"]["english_word_share"] = round(
        sum(w.lower() in EN_CHAT or lid.is_english(w) for w in words) / len(words), 4)
    del norm
    torch.cuda.empty_cache()

    # --- replies ---
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map=dev)
    model = PeftModel.from_pretrained(base, a.adapter)
    model.eval()
    NL = tok("\n", add_special_tokens=False)["input_ids"][-1]
    BEST = dict(do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.2, no_repeat_ngram_size=3)

    @torch.no_grad()
    def reply(ctx, adapter=True, seed=0):
        torch.manual_seed(seed)
        x = tok("".join(f"- {c}\n" for c in ctx) + "- ", return_tensors="pt").to(dev)
        if adapter:
            y = model.generate(**x, max_new_tokens=50, eos_token_id=[tok.eos_token_id, NL],
                               pad_token_id=tok.pad_token_id, **BEST)
        else:
            with model.disable_adapter():
                y = model.generate(**x, max_new_tokens=50, eos_token_id=[tok.eos_token_id, NL],
                                   pad_token_id=tok.pad_token_id, **BEST)
        return tok.decode(y[0][x["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0].strip()

    real_items = []
    for i, (c, (_, te)) in enumerate(zip(comments, und_comments)):
        real_items.append({"message_typed": c, "message": te, "tuned": reply([te], True, i), "base": reply([te], False, i)})
    movie_items = []
    pool = [r["target"] for r in test]
    for i, r in enumerate(test[: a.n_movie]):
        movie_items.append({"message": r["context"][-1], "real": r["target"],
                            "random": rng.choice([p for p in pool if p != r["target"]]),
                            "tuned": reply(r["context"], True, i), "base": reply(r["context"], False, i)})
    GEN.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"understanding": res, "real": real_items, "movie": movie_items}, open(GEN, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, indent=1))
    print("GENERATE_DONE")

else:
    G = json.load(open(GEN, encoding="utf-8"))
    PROMPT = ("You are judging replies in a Telugu conversation. A person said:\n\"{m}\"\n\n"
              "Reply A: \"{a}\"\nReply B: \"{b}\"\n\n"
              "Which reply is a better next line in a natural, spoken Telugu conversation: relevant to what was said, "
              "fluent, and sounding like how people really talk? Answer with a single letter: A or B.")
    pairs = []  # (comparison name, message, first, second)
    for it in G["movie"]:
        pairs += [("real vs random (reliability)", it["message"], it["real"], it["random"]),
                  ("fine-tuned vs random", it["message"], it["tuned"], it["random"]),
                  ("fine-tuned vs real", it["message"], it["tuned"], it["real"]),
                  ("fine-tuned vs base (movie)", it["message"], it["tuned"], it["base"])]
    for it in G["real"]:
        pairs.append(("fine-tuned vs base (real comments)", it["message"], it["tuned"], it["base"]))
    prompts = []
    for name, m, x, y in pairs:
        prompts += [PROMPT.format(m=m, a=x, b=y), PROMPT.format(m=m, a=y, b=x)]

    def parse(t):
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip().upper()
        m = re.search(r"\b([AB])\b", t)
        return m.group(1) if m else "?"

    try:  # vLLM with fp8 weights fits the 24B judge on a 40 GB A100 and batches efficiently
        from vllm import LLM, SamplingParams
        llm = LLM(a.judge, quantization="fp8", max_model_len=2048, gpu_memory_utilization=0.9)
        tok = llm.get_tokenizer()
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for p in prompts]
        outs = [o.outputs[0].text for o in llm.generate(chats, SamplingParams(temperature=0, max_tokens=8))]
        engine = "vLLM fp8"
    except Exception as e:  # fallback: transformers 4-bit
        print("vLLM failed, falling back to transformers 4-bit:", str(e)[:200], flush=True)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        tok = AutoTokenizer.from_pretrained(a.judge, padding_side="left")
        tok.pad_token = tok.pad_token or tok.eos_token
        jm = AutoModelForCausalLM.from_pretrained(a.judge, device_map="cuda", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))
        outs = []
        for i in range(0, len(prompts), 16):
            chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for p in prompts[i:i + 16]]
            x = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            y = jm.generate(**x, max_new_tokens=8, do_sample=False)
            outs += tok.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)
        engine = "transformers 4-bit"

    verdicts = [parse(o) for o in outs]
    results = {}
    for k, (name, *_ ) in enumerate(pairs):
        first, second = verdicts[2 * k], verdicts[2 * k + 1]  # order 1: first=A ; order 2: first=B
        if first == "A" and second == "B":
            w = "first"
        elif first == "B" and second == "A":
            w = "second"
        else:
            w = "tie/inconsistent"
        r = results.setdefault(name, {"first": 0, "second": 0, "tie/inconsistent": 0})
        r[w] += 1
    for name, r in results.items():
        n = sum(r.values())
        r.update({"n": n, "first_win_rate": round(r["first"] / n, 4), "second_win_rate": round(r["second"] / n, 4)})
    # Only statistics go into the repository: no comment text (real people's words stay in data/realworld/, local only)
    out = {"engine": engine, "judge": a.judge, "understanding": G["understanding"], "pairwise": results,
           "unparsed_share": round(verdicts.count("?") / len(verdicts), 4)}
    OUT.parent.mkdir(parents=True, exist_ok=True)  # results folder
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out["pairwise"], indent=1))
    print("JUDGE_DONE")
