"""Try the project's core ideas in a few seconds: no GPU, no dataset, no model weights needed.

    pip install sentencepiece
    python examples/quickstart.py

1. Romanizer: the same Telugu sentence typed many different ways (spelling habits measured from real people)
2. Phonetic key: different spellings collapse to one form
3. Tokenizers: train BPE and Unigram on your own small text and see how they cut romanized Telugu
"""
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlm.romanizer import romanize  # noqa: E402
from tlm.textnorm import phonetic_key  # noqa: E402

SENTENCES = ["ఏంట్రా, ఇంత లేట్ అయ్యింది?", "నువ్వు ఎగ్జామ్ ఎలా రాసావ్?", "రేపు సినిమాకి వెళ్దామా?", "నేను ఇంటికి వెళ్ళిపోతున్నాను"]

print("1. One Telugu sentence, typed six different ways\n")
for s in SENTENCES:
    print(f"   {s}")
    print(f"     canonical : {romanize(s, canonical=True)}")
    print(f"     as typed  : " + "  |  ".join(romanize(s, random.Random(k)) for k in range(5)))
    print()

print("2. Phonetic key: different spellings of the same sentence become (almost) identical\n")
for k in range(4):
    typed = romanize(SENTENCES[0], random.Random(k))
    print(f"   {typed:32} ->  {phonetic_key(typed)}")
print()

print("3. Tokenizers trained on a tiny sample (try it with your own text)\n")
try:
    import sentencepiece as spm
except ImportError:
    sys.exit("   (pip install sentencepiece to run this part)")
rng = random.Random(0)
lines = [romanize(s, random.Random(rng.random())) for s in SENTENCES for _ in range(200)]
with tempfile.TemporaryDirectory() as d:
    f = Path(d) / "text.txt"
    f.write_text("\n".join(lines), encoding="utf-8")
    for kind in ["bpe", "unigram"]:
        spm.SentencePieceTrainer.train(input=str(f), model_prefix=f"{d}/{kind}", vocab_size=60, model_type=kind,
                                       character_coverage=1.0, minloglevel=2)
        sp = spm.SentencePieceProcessor(model_file=f"{d}/{kind}.model")
        for w in ["enti", "yenti", "intha", "inta", "vellipothunnanu"]:
            print(f"   {kind:8} {w:16} -> {' '.join(sp.encode(w, out_type=str))}")
        print()
print("See docs/STAGES.md for what these experiments showed at full scale.")
