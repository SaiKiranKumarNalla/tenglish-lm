"""Stage 1: domain-adapt sarvam-2b to Telugu movie dialogue with LoRA (bf16, L4/A100).

Run on a Colab GPU VM from the repo root:
    python mlm/train_stage1.py --data data/processed --out runs/stage1 [--max_steps 30]

Writes runs/stage1/{adapter/, metrics.json, samples.md}.
Success = test perplexity of the adapted model < base model on the held-out movies.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

MODEL_ID = "sarvamai/sarvam-2b-v0.5"
PROMPTS = [  # fixed qualitative probes: header + first turn, model continues
    "భావోద్వేగం: ELEVATION_MASS | మాండలికం: RAYALASEEMA\n- ",
    "భావోద్వేగం: COMEDY | మాండలికం: TELANGANA\n- ",
    "భావోద్వేగం: EMOTIONAL_HEARTBREAK | మాండలికం: COASTAL_ANDHRA\n- అమ్మా, నేను వెళ్ళిపోతున్నాను.\n- ",
    "భావోద్వేగం: ANGER_WARNING | మాండలికం: STANDARD_TELUGU\n- ",
]


def build_lm_dataset(ds, tok, block):
    """Tokenize scenes (+EOS) and pack them into fixed-length blocks, so no compute is wasted on padding."""
    def tokenize(batch):
        return tok([t + tok.eos_token for t in batch["text"]], add_special_tokens=False)

    def pack(batch):
        ids = sum(batch["input_ids"], [])
        n = (len(ids) // block) * block
        chunks = [ids[i:i + block] for i in range(0, n, block)]
        return {"input_ids": chunks, "attention_mask": [[1] * block for _ in chunks]}

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    return ds.map(pack, batched=True, remove_columns=ds.column_names)


@torch.no_grad()
def perplexity(model, ds, collator, bs=8):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(ds), bs):
        batch = collator([ds[j] for j in range(i, min(i + bs, len(ds)))])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        loss = model(**batch).loss
        k = batch["input_ids"].numel()
        total, n = total + loss.item() * k, n + k
    return math.exp(total / n)


@torch.no_grad()
def generate(model, tok, prompt):
    x = tok(prompt, return_tensors="pt").to(model.device)
    y = model.generate(**x, max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.9,
                       repetition_penalty=1.1, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    return tok.decode(y[0][x["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/stage1")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=-1, help="set e.g. 30 for a smoke test")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    raw = load_dataset("json", data_files={s: f"{a.data}/{s}.jsonl" for s in ["train", "val", "test"]})
    ds = {s: build_lm_dataset(raw[s], tok, a.block) for s in raw}
    print({s: len(d) for s, d in ds.items()}, "blocks of", a.block, "tokens")
    collator = DataCollatorForLanguageModeling(tok, mlm=False)  # causal LM: labels = input_ids

    # Baseline: the same weights with LoRA switched off = untouched sarvam-2b
    with model.disable_adapter():
        base_ppl = perplexity(model, ds["test"], collator)
        base_samples = [generate(model, tok, p) for p in PROMPTS]
    print(f"base test ppl = {base_ppl:.2f}")

    args = TrainingArguments(
        output_dir=str(out / "ckpt"), per_device_train_batch_size=8, gradient_accumulation_steps=2,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.05, num_train_epochs=a.epochs,
        max_steps=a.max_steps, bf16=True, logging_steps=10, eval_strategy="steps", eval_steps=50,
        save_strategy="steps", save_steps=50, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, report_to="none",
        per_device_eval_batch_size=8, gradient_checkpointing=True)
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
                      data_collator=collator, callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    t0 = time.time()
    trainer.train()

    tuned_ppl = perplexity(model, ds["test"], collator)
    tuned_samples = [generate(model, tok, p) for p in PROMPTS]
    model.save_pretrained(out / "adapter")

    metrics = {"base_test_ppl": base_ppl, "tuned_test_ppl": tuned_ppl,
               "train_minutes": (time.time() - t0) / 60, "blocks": {s: len(d) for s, d in ds.items()},
               "log_history": trainer.state.log_history}
    json.dump(metrics, open(out / "metrics.json", "w"), ensure_ascii=False, indent=1)
    with open(out / "samples.md", "w", encoding="utf-8") as f:
        for p, b, t in zip(PROMPTS, base_samples, tuned_samples):
            f.write(f"## Prompt\n```\n{p}\n```\n**Base:**\n```\n{b}\n```\n**Tuned:**\n```\n{t}\n```\n\n")
    print(f"DONE base_ppl={base_ppl:.2f} tuned_ppl={tuned_ppl:.2f}")


if __name__ == "__main__":
    main()
