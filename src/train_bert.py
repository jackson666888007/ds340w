import argparse, time, json
from pathlib import Path
from typing import Dict, List
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback)
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

from src.data_utils import read_text_label_csv, ensure_split_indices, make_splits

def build_ds(texts: List[str], labels: List[int]):
    return Dataset.from_dict({"text": texts, "label": labels})

def main(args):
    texts, labels = read_text_label_csv(args.data_path)
    split = ensure_split_indices(len(texts), args.split_path, seed=args.seed)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = make_splits(texts, labels, split)

    model_name = args.model_name
    tok = AutoTokenizer.from_pretrained(model_name)
    def tok_fn(batch): return tok(batch["text"], truncation=True, padding="max_length", max_length=args.max_len)

    dtr = build_ds(Xtr, Ytr).map(tok_fn, batched=True)
    dva = build_ds(Xva, Yva).map(tok_fn, batched=True)
    dte = build_ds(Xte, Yte).map(tok_fn, batched=True)

    dtr = dtr.remove_columns(["text"]).with_format("torch")
    dva = dva.remove_columns(["text"]).with_format("torch")
    dte = dte.remove_columns(["text"]).with_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = (logits.argmax(-1)).astype(int)
        return {
            "macro_f1": f1_score(labels, preds, average="macro"),
            "accuracy": accuracy_score(labels, preds),
        }

    targs = TrainingArguments(
        output_dir=str(out_dir/"checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=args.fp16,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dtr,
        eval_dataset=dva,
        tokenizer=tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    pred = trainer.predict(dte)
    logits = pred.predictions
    p_pos = (logits[:,1] - logits[:,0])
    y_pred = logits.argmax(-1)
    macro_f1 = f1_score(Yte, y_pred, average="macro")
    acc = accuracy_score(Yte, y_pred)
    cm = confusion_matrix(Yte, y_pred).tolist()

    import torch, timeit
    model.eval()
    with torch.no_grad():
        subset = {k: v[:256] for k,v in dte.with_format("torch").items()}
        def _fwd(): model(**subset)
        infer_latency = timeit.timeit(_fwd, number=20) / (20 * subset["input_ids"].shape[0])

    import csv
    with open(out_dir/"test_preds_bert.csv","w",newline="",encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["p_pos","y_true"])
        import torch.nn.functional as F, torch
        probs = F.softmax(torch.tensor(logits), dim=-1)[:,1].numpy().tolist()
        for p, y in zip(probs, Yte): w.writerow([float(p), int(y)])

    metrics = {
        "model": model_name,
        "macro_f1": float(macro_f1), "accuracy": float(acc),
        "train_time_sec": float(train_time),
        "infer_latency_sec_per_item": float(infer_latency),
        "confusion_matrix": cm,
        "max_len": args.max_len, "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size
    }
    (out_dir/"bert_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir/"bert_report.md").write_text(
        f"# BERT (test)\n- Model: **{model_name}**\n"
        f"- Macro-F1: **{macro_f1:.4f}**  \n- Accuracy: **{acc:.4f}**  \n"
        f"- Train time: {train_time:.2f}s  \n- Inference latency: {infer_latency*1000:.3f} ms/review  \n"
        f"- Confusion matrix: {cm}\n", encoding="utf-8"
    )
    print("Saved metrics & report to", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/IMDB_50k.csv")
    ap.add_argument("--split_path", default="data/split_indices.json")
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=340)
    main(ap.parse_args())
