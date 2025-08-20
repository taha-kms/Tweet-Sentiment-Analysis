# src/dataloaders.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding

import yaml

# ---- local
from .utils import get_tokenizer


# -----------------------------
# config helpers
# -----------------------------
def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _cfg_get(dct, key, default):
    try:
        return dct.get(key, default)
    except Exception:
        return default


# -----------------------------
# simple PyTorch dataset (tokenize on __getitem__)
# -----------------------------
@dataclass
class TweetDataset(TorchDataset):
    df: pd.DataFrame
    tokenizer: PreTrainedTokenizerBase
    text_col: str
    label_col: str
    max_length: int

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        label = int(row[self.label_col])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            # no padding here -> dynamic padding via collator
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# -----------------------------
# HF Datasets path (batched map)
# -----------------------------
def build_hf_dataset_from_df(
    splits: Dict[str, pd.DataFrame],
    tokenizer: PreTrainedTokenizerBase,
    text_col: str,
    label_col: str,
    max_length: int,
) -> DatasetDict:
    def _tok(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=max_length)

    out = {}
    for name, df in splits.items():
        use_df = df[[text_col, label_col]].rename(columns={label_col: "labels"})
        ds = HFDataset.from_pandas(use_df, preserve_index=False)
        ds = ds.map(_tok, batched=True, remove_columns=[text_col])
        ds = ds.with_format("torch", columns=["input_ids", "attention_mask", "labels"])
        out[name] = ds
    return DatasetDict(out)


# -----------------------------
# utils: collator, prints
# -----------------------------
def build_collator(tokenizer: PreTrainedTokenizerBase) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

def _print_label_dist(df: pd.DataFrame, label_col: str, name: str):
    vc = df[label_col].value_counts().sort_index()
    total = int(vc.sum())
    pct = (vc / total * 100).round(2)
    print(f"[{name}] n={total} | label distribution:")
    for lab, cnt in vc.items():
        print(f"  - {lab}: {cnt} ({pct[lab]}%)")

def _peek_batch(dl: DataLoader, name: str):
    batch = next(iter(dl))
    x_ids, x_mask, y = batch["input_ids"], batch["attention_mask"], batch["labels"]
    print(f"[{name} batch] input_ids: {tuple(x_ids.shape)}, attention_mask: {tuple(x_mask.shape)}, labels: {tuple(y.shape)}")


# -----------------------------
# main: load parquet -> tokenize -> sanity
# -----------------------------
def main():
    cfg = _load_yaml("configs/config.yaml")  # uses your config keys :contentReference[oaicite:2]{index=2}

    text_col = cfg["data"]["text_col"]            # 'text' :contentReference[oaicite:3]{index=3}
    label_col = cfg["data"]["label_col"]          # 'label' :contentReference[oaicite:4]{index=4}
    model_name = cfg["model"]["name"]             # e.g., 'bert-base-uncased' :contentReference[oaicite:5]{index=5}
    max_length = cfg["model"]["max_length"]       # e.g., 128 :contentReference[oaicite:6]{index=6}
    train_bs   = _cfg_get(cfg.get("train", {}), "batch_size", 32)         # :contentReference[oaicite:7]{index=7}
    eval_bs    = _cfg_get(cfg.get("train", {}), "eval_batch_size", 64)    # :contentReference[oaicite:8]{index=8}

    proc_dir = cfg["paths"]["proc_dir"]           # 'data/processed' :contentReference[oaicite:9]{index=9}
    train_fp = os.path.join(proc_dir, "train.parquet")
    val_fp   = os.path.join(proc_dir, "val.parquet")
    test_fp  = os.path.join(proc_dir, "test.parquet")

    # Load existing parquet artifacts (produced by your dataset.py)
    train_df = pd.read_parquet(train_fp)
    val_df   = pd.read_parquet(val_fp)
    test_df  = pd.read_parquet(test_fp)

    # Label distribution sanity
    _print_label_dist(train_df, label_col, "train")
    _print_label_dist(val_df,   label_col, "validation")
    _print_label_dist(test_df,  label_col, "test")

    # Tokenizer + dynamic padding collator
    tokenizer = get_tokenizer(model_name)
    collator  = build_collator(tokenizer)

    # ---- PyTorch path
    train_ds_pt = TweetDataset(train_df, tokenizer, text_col, label_col, max_length)
    val_ds_pt   = TweetDataset(val_df,   tokenizer, text_col, label_col, max_length)
    test_ds_pt  = TweetDataset(test_df,  tokenizer, text_col, label_col, max_length)

    train_dl_pt = DataLoader(train_ds_pt, batch_size=train_bs, shuffle=True,  collate_fn=collator)
    val_dl_pt   = DataLoader(val_ds_pt,   batch_size=eval_bs,  shuffle=False, collate_fn=collator)
    test_dl_pt  = DataLoader(test_ds_pt,  batch_size=eval_bs,  shuffle=False, collate_fn=collator)

    _peek_batch(train_dl_pt, "PyTorch/train")
    _peek_batch(val_dl_pt,   "PyTorch/val")

    # ---- HF Datasets path (optional)
    ds_dict = build_hf_dataset_from_df(
        {"train": train_df, "validation": val_df, "test": test_df},
        tokenizer,
        text_col=text_col,
        label_col=label_col,
        max_length=max_length,
    )
    train_dl_hf = DataLoader(ds_dict["train"], batch_size=train_bs, shuffle=True, collate_fn=collator)
    _peek_batch(train_dl_hf, "HF/train")


if __name__ == "__main__":
    main()
