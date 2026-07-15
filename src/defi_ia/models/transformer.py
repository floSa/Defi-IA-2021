"""Transformer fine-tuning for 28-way job classification (Step C).

Runs on GPU only (Kaggle T4 / RTX 4060 Ti). Kept out of the core CPU
environment; install with ``pip install -r requirements-dl.txt``.

Design choices for a 2026 state-of-the-art take that beats a 2021 BERT:

* **Backbone** is configurable via ``model_name`` — target ``ModernBERT-base``
  (efficient 2024 encoder) and ``microsoft/deberta-v3-base`` (strong baseline),
  with ``-large`` variants when VRAM allows.
* **Class-weighted cross-entropy** (inverse-frequency, normalised) so the rare
  0.4 % classes drive Macro-F1 as much as ``professor`` — the same lever that
  mattered for the linear model.
* **Early stopping on validation Macro-F1**, the actual competition metric,
  rather than loss.
* **fp16** mixed precision; gradient accumulation lets an 8 GB card emulate the
  T4's effective batch size.

The heavy imports (torch/transformers) are done lazily inside the functions so
that importing this module on a CPU-only box (e.g. for linting) does not fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TransformerConfig:
    # roberta-base is the proven-stable default; DeBERTa-v3 NaN-diverges on this
    # stack (see reports/ROADMAP.md §2). fp16 is fine for RoBERTa.
    model_name: str = "roberta-base"
    max_length: int = 192
    batch_size: int = 32
    grad_accum: int = 1
    learning_rate: float = 2e-5
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    class_weighted_loss: bool = True
    early_stopping_patience: int = 2
    seed: int = 42
    output_dir: str = "models/transformer"
    num_labels: int = 28
    extra: dict = field(default_factory=dict)


def compute_class_weights(labels: np.ndarray, num_labels: int) -> np.ndarray:
    """Inverse-frequency weights normalised to mean 1 (stable LR)."""
    counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_labels * counts)
    return (weights / weights.mean()).astype(np.float32)


def _build_trainer(cfg, train_df, valid_df):
    """Assemble a HF ``Trainer`` (imports torch/transformers lazily)."""
    import torch
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

    def to_ds(df):
        ds = Dataset.from_dict({"text": df["text"].tolist(),
                                "labels": df["Category"].astype(int).tolist()})
        return ds.map(tok, batched=True, remove_columns=["text"])

    train_ds, valid_ds = to_ds(train_df), to_ds(valid_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels
    )

    class_weights = None
    if cfg.class_weighted_loss:
        w = compute_class_weights(train_df["Category"].to_numpy(), cfg.num_labels)
        class_weights = torch.tensor(w)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            weight = class_weights.to(outputs.logits.device) if class_weights is not None else None
            loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels, weight=weight)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"macro_f1": f1_score(labels, preds, average="macro")}

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        max_grad_norm=1.0,
        # Eval per epoch, not per-N-steps: step-wise early stopping killed
        # training at ~0.13 epoch when Macro-F1 is still ~0 during warmup.
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=100,
        report_to="none",
        seed=cfg.seed,
    )

    return WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    ), tokenizer


def fine_tune(cfg: TransformerConfig, train_df, valid_df):
    """Fine-tune and return ``(trainer, tokenizer)``. GPU required."""
    trainer, tokenizer = _build_trainer(cfg, train_df, valid_df)
    trainer.train()
    return trainer, tokenizer


def predict_logits(trainer, tokenizer, texts, max_length: int, batch_size: int = 64):
    """Return raw logits ``(n, 28)`` for a list of texts (for ensembling)."""
    from datasets import Dataset

    ds = Dataset.from_dict({"text": list(texts)})
    ds = ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )
    return trainer.predict(ds).predictions
