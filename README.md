
# 🐦 Tweet Sentiment Analysis

A PyTorch + HuggingFace project for **sentiment classification on tweets** (Negative / Neutral / Positive).
It supports multiple datasets, handles label canonicalization, trains transformer-based models (BERT, RoBERTa, BERTweet), and produces detailed evaluation reports.

---

## 📂 Project Structure

```
configs/
  config.yaml        # Main configuration for data, model, training
notebooks/
  exploration.ipynb  # Data exploration & visualization
src/
  dataset.py         # Preprocess raw datasets → processed parquet splits
  dataloaders.py     # Torch/HF dataloaders (sanity check script)
  train.py           # Train a transformer model
  evaluate.py        # Evaluate a trained model
  utils.py           # Shared helpers
models/              # (ignored) trained models, checkpoints, eval results
runs/                # (ignored) TensorBoard logs
data/
  raw/               # Put your raw datasets here (CSV)
  processed/         # Auto-generated processed parquet splits
```

---

## ⚙️ Requirements

* Python 3.8+
* PyTorch
* HuggingFace Transformers
* Pandas, Numpy, Scikit-learn
* TensorBoard

Install everything:

```bash
pip install -r requirements.txt
```

---

## 📊 Datasets

This project integrates **three Kaggle datasets** for training/evaluation:

1. **[Twitter Sentiment Dataset](https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset)**

   * Labels: `-1` = Negative, `0` = Neutral, `+1` = Positive
   * Canonical mapping: `{ -1 → 0, 0 → 1, 1 → 2 }`

2. **[Twitter Tweets Sentiment Dataset](https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset)**

   * Labels: `"negative"`, `"neutra"`, `"neutral"`, `"positive"`
   * Canonical mapping: `{ negative → 0, neutra → 1, neutral → 1, positive → 2 }`

3. **[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)**

   * Labels: `0` = Negative, `2` = Neutral, `4` = Positive
   * Canonical mapping: `{ 0 → 0, 2 → 1, 4 → 2 }`

📌 These mappings are already defined in [`configs/config.yaml`](configs/config.yaml).

---

## 🚀 Usage

### 1. Preprocess datasets

```bash
python3 -m src.dataset --config configs/config.yaml
```

Builds merged & split parquet files under `data/processed/`.

### 2. (Optional) Inspect dataloaders

```bash
python3 -m src.dataloaders
```

### 3. Train a model

```bash
python3 -m src.train --config configs/config.yaml
```

### 4. Evaluate models

```bash
python3 -m src.evaluate --last --include_plots
```

---

## 📝 Configuration (`configs/config.yaml`)

See [config file](configs/config.yaml) for:

* Paths (data, models, logs)
* Cleaning rules
* Label mappings
* Train/val/test splits
* Model choice & hyperparams

---

## 📒 Notebook

[`notebooks/exploration.ipynb`](notebooks/exploration.ipynb) for:

* Distribution checks
* Class balance
* Word clouds
* Exploratory analysis

---

# ⚙️ CLI & Arguments

This project exposes three main entrypoints:

* **Dataset builder:** `python3 -m src.dataset`
* **Training:** `python3 -m src.train`
* **Evaluation:** `python3 -m src.evaluate`

Below are the important flags and how to use them.

---

## 📦 `src.dataset` — build processed splits

```bash
python3 -m src.dataset --config configs/config.yaml [--no-split]
```

**Args**

* `--config` Path to YAML config (paths, cleaning, label mapping, split sizes).&#x20;
* `--no-split` Only create `merged.parquet` (skip train/val/test).&#x20;

**Outputs**

* `data/processed/merged.parquet` + `train.parquet`, `val.parquet`, `test.parquet`.&#x20;
* A small `runs/data_report.json` summary.&#x20;

---

## 🏋️ `src.train` — train a transformer model

```bash
python3 -m src.train --config configs/config.yaml \
                     [--models "bert-base-uncased,roberta-base"] \
                     [--models_root models] \
                     [--train_path data/processed/train.parquet] \
                     [--val_path data/processed/val.parquet] \
                     [--device cuda|cpu]
```

**Args**

* `--config` Config to load hyperparams, paths, etc.&#x20;
* `--models` Comma-separated HF model names; if omitted uses `model.name` from the config.&#x20;
* `--models_root` Parent folder where runs are saved (defaults to `paths.models_dir`).&#x20;
* `--train_path`, `--val_path` Parquet splits to use.&#x20;
* `--device` `cuda` (default if available) or `cpu`.&#x20;

**Run layout**

```
models/<MODEL>_<YYYYMMDD-HHMM>/
  best/         # best epoch by metric (default: f1_macro)
  last/         # last epoch
  tokenizer/
  train_log.csv # per-epoch metrics
  params.json   # (config + meta) recorded for provenance
```



---

## ✅ `src.evaluate` — evaluate checkpoints & keep a global index

```bash
# Evaluate newest trained model (auto-detected), prefer BEST checkpoint
python3 -m src.evaluate --last [--include_plots] [--run_tag some-note]

# Re-evaluate a specific run folder (auto-picks best/ inside it if present)
python3 -m src.evaluate --checkpoint models/bert-base-uncased_20250820-1538 [--include_plots]

# Evaluate an explicit subfolder (force BEST or LAST)
python3 -m src.evaluate --checkpoint models/bert-base-uncased_20250820-1538/best
python3 -m src.evaluate --checkpoint models/bert-base-uncased_20250820-1538/last
```

**Key args**

* `--last`
  Picks the **most recent** run directory under `<models_root>` and evaluates it (prefers `best/`, falls back to `last/`). Great for “just trained something; now evaluate it.”&#x20;
* `--checkpoint PATH`
  Re-evaluate a **specific model**. `PATH` can be the run folder (script will prefer `best/`), or a direct path to `best/` or `last/`. Use this when you want to compare runs or regenerate plots.&#x20;
* `--run_tag TAG` (**tags**)
  Adds a short note to the eval folder name so you can tell runs apart, e.g. `eval/20250820-1802_paperA`.&#x20;
* `--include_plots`
  Saves reliability and ROC-AUC plots, plus confusion matrix image.&#x20;
* `--batch_size`, `--num_workers`, `--device`, `--val_path`, `--test_path`
  Control evaluation speed, device and which splits to score.&#x20;

**Eval outputs (per run)**

```
models/<RUN>/eval/<YYYYMMDD-HHMM[_TAG]>/
  metrics_val.json
  metrics_test.json
  metrics_overview.csv    # val/test accuracy + F1 macro/micro table
  misclassified_val.csv
  misclassified_test.csv
  hard_cases.csv          # top-100 highest-loss examples across val+test
  confusion_matrix.png    # + optional reliability / ROC-AUC plots
```



---

## 📒 Global leaderboard: `overall_evaluations.csv`

After **each** evaluation, one row is appended to:
`<models_root>/overall_evaluations.csv`.&#x20;

**What’s recorded per row**

* `model_run` — the run folder name, e.g. `bert-base-uncased_20250820-1538`.
* `eval_dir` — the exact eval folder path.
* `val_accuracy`, `val_f1_macro`, `val_f1_micro`, `test_accuracy`, `test_f1_macro`, `test_f1_micro`.&#x20;
* **All parameters flattened** from `params.json` (both `config.*` and `meta.*`) so each becomes its own column, enabling easy slicing/filtering in pandas/Sheets.&#x20;

This lets you build a simple **leaderboard** across runs (filter by model name, max length, seed, etc.) without opening each eval folder.

> Tip: If you later decide to record *every fine-grained metric* (e.g., per-class precision/recall, per-source scores) into the global file, you can extend the script by flattening the full `val_metrics`/`test_metrics` dict before appending (your code already has a `flatten()` helper).&#x20;

---

## 🧠 Common scenarios

* **“Give me the newest model’s scores”**
  `python3 -m src.evaluate --last`  → picks latest run, evaluates `best/`.&#x20;

* **“Re-evaluate this exact model and keep results separate with a tag”**
  `python3 -m src.evaluate --checkpoint models/<RUN> --run_tag recheck --include_plots`
  Creates `models/<RUN>/eval/<STAMP>_recheck/…`.&#x20;

* **“Compare multiple trainings in a sheet”**
  Open `<models_root>/overall_evaluations.csv` — each eval is one row with metrics + config/meta columns.&#x20;

---


