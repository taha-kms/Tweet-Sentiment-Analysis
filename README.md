
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

## 🗂️ Git Hygiene

Keep heavy outputs out of Git. `.gitignore` should include:

* `data/raw/`, `data/processed/`
* `models/`, `runs/`
* `*.log`, `.ipynb_checkpoints/`

---

## ✅ Workflow Summary

```bash
# 1. Prepare dataset
python3 -m src.dataset --config configs/config.yaml

# 2. Check dataloaders
python3 -m src.dataloaders

# 3. Train model
python3 -m src.train --config configs/config.yaml

# 4. Evaluate best checkpoint
python3 -m src.evaluate --last --include_plots
```

---

