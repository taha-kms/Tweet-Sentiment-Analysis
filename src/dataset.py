# src/dataset.py
from __future__ import annotations

import os
import re
import argparse
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import load_config, get_logger, set_seed, save_json, _ensure_dir


# =============================
# Helpers for config/object coercion
# =============================
def _as_dict(obj):
    """Recursively convert SimpleNamespace → dict (leave dicts as-is)."""
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: _as_dict(v) for k, v in obj.__dict__.items()}
    return obj

def _cfg_get(section, key: str, default):
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


# =============================
# Cleaning regexes
# =============================
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")
SMART_QUOTES_RE = re.compile(r"[“”]")

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


# =============================
# CSV reading (defensive)
# =============================
def _read_csv_safely(path: str) -> pd.DataFrame:
    """
    Defensive CSV read: texts may contain commas/quotes/newlines.
    """
    return pd.read_csv(
        path,
        sep=",",
        engine="python",          # tolerant parser
        quoting=0,                # csv.QUOTE_MINIMAL
        escapechar="\\",
        on_bad_lines="skip",
    )


# =============================
# Text cleaning
# =============================
def _clean_text(
    s: Any,
    *,
    lower: bool,
    replace_urls: bool,
    replace_mentions: bool,
    keep_emojis: bool,
) -> str:
    if not isinstance(s, str):
        return ""
    t = SMART_QUOTES_RE.sub('"', s)
    if replace_urls:
        t = URL_RE.sub("<URL>", t)
    if replace_mentions:
        t = MENTION_RE.sub("<USER>", t)
    if not keep_emojis:
        # crude emoji stripping: drop non-BMP codepoints
        t = "".join(ch for ch in t if ord(ch) <= 0xFFFF)
    t = WHITESPACE_RE.sub(" ", t).strip()
    if lower:
        t = t.lower()
    return t


# =============================
# Label mapping (per source)
# =============================
def _norm_label_key(v: Any) -> str:
    """
    Normalize raw label value to a canonical string key:
    - strip quotes/whitespace
    - lowercase strings
    - '1.0' -> '1' (float-like ints collapse to ints)
    - treat missing-like as '___MISSING___'
    """
    if v is None:
        return "___MISSING___"
    s = str(v).strip().strip('"').strip("'")
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return "___MISSING___"
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return s
    except Exception:
        return s.lower()


def _get_value_to_canonical_map(source_name: str, cfg) -> Dict[str, int]:
    """
    Return dict: normalized_raw_value -> canonical_id {0,1,2}.
    Priority:
      1) exact key in cfg.data.value_to_canonical
      2) basename(source).lower() key
      3) DEFAULT mapping for unknown files: -1→0, 0→1, 1→2
    """
    # pull value_to_canonical and coerce to plain dict
    if isinstance(cfg.data, dict):
        vt = cfg.data.get("value_to_canonical")
    else:
        vt = getattr(cfg.data, "value_to_canonical", None)
    vt = _as_dict(vt) if vt is not None else None

    def _normalize_keys(d):
        return {_norm_label_key(k): int(v) for k, v in d.items()}

    # exact filename key (as written in config)
    if isinstance(vt, dict) and source_name in vt:
        return _normalize_keys(vt[source_name])

    # basename lowercase
    key = os.path.basename(source_name).lower()
    if isinstance(vt, dict) and key in vt:
        return _normalize_keys(vt[key])

    # ---------- DEFAULT FALLBACK FOR UNLISTED FILES ----------
    # You asked for default mapping: raw labels -1, 0, 1 → canonical 0, 1, 2
    return {
        _norm_label_key("-1"): 0,  # negative
        _norm_label_key("0"):  1,  # neutral
        _norm_label_key("1"):  2,  # positive
        # allow common text aliases too (helps zero-config files)
        "negative": 0,
        "neutral": 1,
        "neutra": 1,
        "positive": 2,
    }


def _apply_label_mapping(merged_raw: pd.DataFrame, cfg, logger) -> pd.DataFrame:
    """
    Apply per-source mapping; drop rows with unknown labels and warn.
    """
    parts: List[pd.DataFrame] = []
    for src, chunk in merged_raw.groupby("source_ds", dropna=False):
        mapper = _get_value_to_canonical_map(str(src), cfg)
        raw_norm = chunk["label"].apply(_norm_label_key)
        mapped = raw_norm.map(mapper)

        mask_unknown = mapped.isna()
        if mask_unknown.any():
            unknown_vals = sorted(set(raw_norm[mask_unknown]) - {"___MISSING___"})
            dropped = int(mask_unknown.sum())
            if unknown_vals:
                logger.warning(f"[{src}] Unmapped label values: {unknown_vals}. Dropping {dropped} rows.")
            else:
                logger.warning(f"[{src}] Missing/empty labels. Dropping {dropped} rows.")

        tmp = chunk.loc[~mask_unknown].copy()
        tmp["label"] = mapped[~mask_unknown].astype(int)
        parts.append(tmp)

    if not parts:
        raise RuntimeError("After label mapping, no data remains. Check your mappings or defaults.")
    return pd.concat(parts, axis=0, ignore_index=True)


# =============================
# Dataset-specific / generic loaders
# =============================
TEXT_CANDIDATES = [
    "text", "tweet", "content", "body", "message", "clean_text",
]
LABEL_CANDIDATES = [
    "sentiment", "category", "label", "polarity", "target", "class",
]

def _norm_cols(df: pd.DataFrame) -> Dict[str, str]:
    """Lowercase->original column map."""
    return {c.lower(): c for c in df.columns}

def load_ds1(path: str) -> pd.DataFrame:
    df = _read_csv_safely(path)
    cols = _norm_cols(df)
    text_col = cols.get("tweet", cols.get("text", list(df.columns)[-1]))
    label_col = cols.get("sentiment", cols.get("label", list(df.columns)[0]))
    out = pd.DataFrame({"text": df[text_col], "label": df[label_col]})
    out["source_ds"] = os.path.basename(path)
    return out

def load_ds2(path: str) -> pd.DataFrame:
    df = _read_csv_safely(path)
    cols = _norm_cols(df)
    text_col = cols.get("text", cols.get("tweet", list(df.columns)[1]))
    label_col = cols.get("sentiment", cols.get("category", cols.get("label", list(df.columns)[-1])))
    out = pd.DataFrame({"text": df[text_col], "label": df[label_col]})
    out["source_ds"] = os.path.basename(path)
    return out

def load_ds3(path: str) -> pd.DataFrame:
    df = _read_csv_safely(path)
    cols = _norm_cols(df)
    text_col = cols.get("text", list(df.columns)[0])
    label_col = cols.get("category", cols.get("label", list(df.columns)[1]))
    out = pd.DataFrame({"text": df[text_col], "label": df[label_col]})
    out["source_ds"] = os.path.basename(path)
    return out

def load_generic(path: str, logger) -> pd.DataFrame:
    """
    Generic loader for arbitrary CSVs:
    - pick a reasonable text column
    - pick a reasonable label column (or warn and skip file if none)
    """
    df = _read_csv_safely(path)
    cols = _norm_cols(df)

    # pick text
    text_col = None
    for c in TEXT_CANDIDATES:
        if c in cols:
            text_col = cols[c]
            break
    if text_col is None:
        # fallback: closest guess = longest string-ish column
        text_col = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())

    # pick label
    label_col = None
    for c in LABEL_CANDIDATES:
        if c in cols:
            label_col = cols[c]
            break

    if label_col is None:
        logger.warning(f"[{os.path.basename(path)}] No label column found. Skipping this file.")
        return pd.DataFrame(columns=["text", "label", "source_ds"])

    out = pd.DataFrame({"text": df[text_col], "label": df[label_col]})
    out["source_ds"] = os.path.basename(path)
    return out


def _pick_loader(filename: str):
    """
    Heuristic loader selection by filename.
    """
    name = filename.lower()
    if "ds1" in name:
        return load_ds1
    if "ds2" in name:
        return load_ds2
    if "ds3" in name:
        return load_ds3
    return None  # generic


# =============================
# Build merged
# =============================
def _apply_cleaning_df(df: pd.DataFrame, cfg) -> pd.DataFrame:
    c = cfg.data["clean"] if isinstance(cfg.data, dict) else cfg.data.clean
    lower = _cfg_get(c, "lower", False)
    replace_urls = _cfg_get(c, "replace_urls", True)
    replace_mentions = _cfg_get(c, "replace_mentions", True)
    keep_emojis = _cfg_get(c, "keep_emojis", True)

    cleaned = df.copy()
    cleaned["text"] = (
        cleaned["text"].astype(str).map(
            lambda s: _clean_text(
                s,
                lower=lower,
                replace_urls=replace_urls,
                replace_mentions=replace_mentions,
                keep_emojis=keep_emojis,
            )
        )
    )
    cleaned = cleaned[cleaned["text"].str.len() > 0]
    return cleaned


def _discover_sources(cfg, logger) -> List[str]:
    """
    If data.sources is provided and non-empty, use it.
    Otherwise, discover all *.csv in paths.raw_dir (sorted).
    """
    raw_dir = cfg.paths.raw_dir
    if isinstance(cfg.data, dict):
        sources = cfg.data.get("sources") or []
    else:
        sources = getattr(cfg.data, "sources", []) or []

    if sources:
        return sources

    # auto-discover
    logger.info("No data.sources specified in config; discovering all *.csv in data/raw ...")
    all_csv = [f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")]
    all_csv.sort()
    if not all_csv:
        raise RuntimeError(f"No CSV files found in {raw_dir}")
    return all_csv


def build_merged(cfg, logger) -> pd.DataFrame:
    raw_dir = cfg.paths.raw_dir
    sources: List[str] = _discover_sources(cfg, logger)

    frames: List[pd.DataFrame] = []
    for filename in sources:
        path = os.path.join(raw_dir, filename)
        if not os.path.isfile(path):
            logger.warning(f"Missing dataset file: {path}")
            continue

        loader = _pick_loader(filename)
        if loader is None:
            logger.info(f"Loading {filename} with load_generic ...")
            df = load_generic(path, logger)
        else:
            logger.info(f"Loading {filename} with {loader.__name__} ...")
            df = loader(path)

        # skip files with no labels (generic loader may return empty)
        if df.empty:
            continue

        frames.append(df)

    if not frames:
        raise RuntimeError("No datasets loaded. Check data/raw and your files (labels missing?).")

    merged_raw = pd.concat(frames, axis=0, ignore_index=True)

    # Map raw labels -> canonical {0,1,2}
    merged = _apply_label_mapping(merged_raw, cfg, logger)

    # Clean text
    merged = _apply_cleaning_df(merged, cfg)

    # Deduplicate exact text/label pairs (helps reduce leakage/noise)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    if len(merged) != before:
        logger.info(f"Deduplicated rows: {before - len(merged)}")

    logger.info(f"Merged size: {len(merged)} rows across {merged['source_ds'].nunique()} sources")

    # Pretty per-source label report
    try:
        per_src = merged.groupby(["source_ds", "label"]).size().unstack(fill_value=0)
        per_src = per_src.rename(columns=LABEL_NAMES)
        logger.info("Per-source label counts (canonical):\n" + per_src.to_string())
    except Exception:
        pass

    return merged


# =============================
# Splitting
# =============================
def make_splits(
    merged: pd.DataFrame, cfg, logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_cfg = cfg.data["split"] if isinstance(cfg.data, dict) else cfg.data.split
    seed = _cfg_get(split_cfg, "seed", 42)
    test_size = _cfg_get(split_cfg, "test_size", 0.15)
    val_size = _cfg_get(split_cfg, "val_size", 0.15)
    strategy = _cfg_get(split_cfg, "strategy", "stratified")

    def _maybe_stratify(df: pd.DataFrame) -> Any:
        return df["label"] if df["label"].nunique() > 1 else None

    if strategy == "grouped_by_source":
        counts = merged["source_ds"].value_counts()
        holdout = counts.idxmax()
        test_df = merged[merged["source_ds"] == holdout]
        rest = merged[merged["source_ds"] != holdout]
        train_df, val_df = train_test_split(
            rest,
            test_size=val_size,
            random_state=seed,
            stratify=_maybe_stratify(rest),
        )
        logger.info(
            f"Grouped split: test source={holdout} | "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
        return train_df, val_df, test_df

    # default: stratified random split (when possible)
    train_val, test_df = train_test_split(
        merged,
        test_size=test_size,
        random_state=seed,
        stratify=_maybe_stratify(merged),
    )
    val_rel = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val,
        test_size=val_rel,
        random_state=seed,
        stratify=_maybe_stratify(train_val),
    )
    logger.info(
        f"Stratified split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return train_df, val_df, test_df


# =============================
# CLI
# =============================
def main():
    parser = argparse.ArgumentParser(description="Build and split datasets for BERT sentiment.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--no-split", action="store_true", help="Only build merged.parquet")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_level = getattr(cfg.logging, "log_level", "INFO") if hasattr(cfg, "logging") else "INFO"
    logger = get_logger(level=log_level)
    split_cfg = cfg.data["split"] if isinstance(cfg.data, dict) else cfg.data.split
    seed = _cfg_get(split_cfg, "seed", 42)
    set_seed(seed)

    out_dir = cfg.paths.proc_dir
    _ensure_dir(out_dir)
    _ensure_dir(cfg.paths.runs_dir)

    # Build merged
    merged = build_merged(cfg, logger)

    # Save merged + report
    merged_path = os.path.join(out_dir, "merged.parquet")
    merged.to_parquet(merged_path, index=False)
    logger.info(f"Saved merged dataset to {merged_path}")

    report = {
        "total_rows": int(len(merged)),
        "by_label": {LABEL_NAMES[int(k)]: int(v) for k, v in merged["label"].value_counts(sort=False).to_dict().items()},
        "by_source": {str(k): int(v) for k, v in merged["source_ds"].value_counts().to_dict().items()},
        "avg_len": float(merged["text"].str.len().mean()),
    }
    save_json(report, os.path.join(cfg.paths.runs_dir, "data_report.json"))
    logger.info(f"Data report written to {os.path.join(cfg.paths.runs_dir, 'data_report.json')}")

    if args.no_split:
        return

    # Splits
    train_df, val_df, test_df = make_splits(merged, cfg, logger)
    train_df.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)
    logger.info(f"Wrote splits to {out_dir}")


if __name__ == "__main__":
    main()
