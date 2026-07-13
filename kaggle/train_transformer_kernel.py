"""Self-contained Kaggle kernel: fine-tune a transformer for the Defi IA task.

Writes to /kaggle/working: submission.csv, test_logits.npy, valid_metrics.json.
Standalone (no defi_ia import). Mirrors src/defi_ia/models/transformer.py.
"""

import glob
import json
import os
import subprocess
import sys

# ModernBERT needs transformers>=4.48; ensure it whether run via API or notebook.
# Only add sentencepiece (DeBERTa-v3 tokenizer). Do NOT upgrade torch or
# transformers: newer torch wheels drop P100 (sm_60) support and Kaggle may
# assign a P100, causing "no kernel image is available for execution".
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "protobuf"], check=False)

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ---- knobs -----------------------------------------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/deberta-v3-base")
MAX_LENGTH = 256
BATCH_SIZE = 8
GRAD_ACCUM = 2
LR = 1e-5
EPOCHS = 2
SEED = 42
FULL_TRAIN = False
VALID_SIZE = 0.15
NUM_LABELS = 28
OUT = "/kaggle/working"


def find_input():
    print("/kaggle/input contents:", os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "MISSING")
    hits = glob.glob("/kaggle/input/**/train.json", recursive=True)
    if not hits:
        raise FileNotFoundError("train.json not found under /kaggle/input; is the competition data attached?")
    d = os.path.dirname(hits[0])
    print("using INPUT =", d)
    return d


def load():
    INPUT = find_input()
    train = pd.read_json(f"{INPUT}/train.json").set_index("Id")
    labels = pd.read_csv(f"{INPUT}/train_label.csv", index_col="Id")["Category"]
    names = pd.read_csv(f"{INPUT}/categories_string.csv")
    id2name = dict(zip(names["1"].astype(int), names["0"], strict=True))
    train["Category"] = labels
    test = pd.read_json(f"{INPUT}/test.json").set_index("Id")
    for df in (train, test):
        df["text"] = df["description"].str.strip().str.replace(r"\s+", " ", regex=True)
    return train, test, id2name


def class_weights(y):
    counts = np.bincount(y, minlength=NUM_LABELS).astype(np.float64)
    counts[counts == 0] = 1.0
    w = np.sqrt(counts.sum() / (NUM_LABELS * counts))
    w = w / w.mean()
    w = np.clip(w, 0.5, 5.0)
    return torch.tensor(w.astype(np.float32))


def macro_disparate_impact(jobs, genders):
    people = pd.DataFrame({"job": list(jobs), "gender": list(genders)})
    c = people.groupby(["job", "gender"]).size().unstack("gender")
    di = c[["M", "F"]].max(axis=1) / c[["M", "F"]].min(axis=1)
    return float(di.mean())


def main():
    train, test, id2name = load()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    weights = class_weights(train["Category"].to_numpy())

    if FULL_TRAIN:
        tr = train
        va = train.sample(frac=0.02, random_state=SEED)
    else:
        tr, va = train_test_split(train, test_size=VALID_SIZE, random_state=SEED, stratify=train["Category"])

    def make_ds(df, with_labels=True):
        d = {"text": df["text"].tolist()}
        if with_labels:
            d["labels"] = df["Category"].astype(int).tolist()
        ds = Dataset.from_dict(d)
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH), batched=True, remove_columns=["text"])

    train_ds, valid_ds = make_ds(tr), make_ds(va)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits.float(), labels, weight=weights.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def metrics(eval_pred):
        logits, labels = eval_pred
        return {"macro_f1": f1_score(labels, np.argmax(logits, -1), average="macro")}

    args = TrainingArguments(
        output_dir=f"{OUT}/ckpt", per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2, gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR, num_train_epochs=EPOCHS, weight_decay=0.01, warmup_ratio=0.1, max_grad_norm=1.0,
        fp16=False, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, save_total_limit=1, logging_steps=100, report_to="none", seed=SEED,
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds,
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics,
    )
    trainer.train()

    val_logits = trainer.predict(valid_ds).predictions
    val_pred = np.argmax(val_logits, -1)
    report = {
        "model": MODEL_NAME,
        "macro_f1": float(f1_score(va["Category"], val_pred, average="macro")),
        "disparate_impact": macro_disparate_impact([id2name[c] for c in val_pred], va["gender"]),
        "full_train": FULL_TRAIN,
    }
    with open(f"{OUT}/valid_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print("VALID METRICS:", report)

    test_ds = make_ds(test, with_labels=False)
    test_logits = trainer.predict(test_ds).predictions
    np.save(f"{OUT}/test_logits.npy", test_logits)
    sub = pd.DataFrame({"Id": test.index, "Category": np.argmax(test_logits, -1)}).sort_values("Id")
    sub.to_csv(f"{OUT}/submission.csv", index=False)
    print("Wrote submission.csv:", sub.shape)


if __name__ == "__main__":
    main()
