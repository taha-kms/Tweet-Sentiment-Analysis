# src/utils.py
from __future__ import annotations

import os
import sys
import json
import time
import yaml
import math
import random
import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional


# -----------------------------
# Small helpers
# -----------------------------
def _to_namespace(obj: Any) -> Any:
    """
    Recursively convert nested dicts to SimpleNamespace for dot access.
    Lists/tuples are preserved with recursive conversion.
    """
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_namespace(v) for v in obj)
    return obj


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _merge_dict(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Shallow+recursive merge: values in 'override' take precedence.
    """
    if not override:
        return base
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


# -----------------------------
# Config loader
# -----------------------------
def load_config(
    path: str,
    overrides: Optional[Dict[str, Any]] = None,
    create_dirs: bool = True,
    return_namespace: bool = True,
) -> SimpleNamespace | Dict[str, Any]:
    """
    Load a YAML config, optionally apply overrides, and (optionally) create directories.

    Args:
        path: Path to YAML (e.g., 'configs/config.yaml').
        overrides: Dict of values to override (same structure as YAML).
        create_dirs: If True, ensure paths under config.paths/* exist.
        return_namespace: If True, return a SimpleNamespace; otherwise a plain dict.

    Returns:
        Config as SimpleNamespace (dot-access) or dict.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f) or {}

    cfg_dict = _merge_dict(cfg_dict, overrides)

    # Normalize and optionally create project directories
    paths = cfg_dict.get("paths", {})
    for key in ("raw_dir", "proc_dir", "models_dir", "runs_dir"):
        if key in paths and isinstance(paths[key], str):
            # normalize path to POSIX-ish form relative to repo root
            paths[key] = os.path.normpath(paths[key])
            if create_dirs:
                _ensure_dir(paths[key])
    cfg_dict["paths"] = paths

    return _to_namespace(cfg_dict) if return_namespace else cfg_dict


# -----------------------------
# Seed setting
# -----------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set seeds across Python, NumPy, and (if available) PyTorch to maximize reproducibility.

    Args:
        seed: Seed value.
        deterministic: If True, enables deterministic behavior in PyTorch (when available).
    """
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np  # noqa: WPS433
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch  # noqa: WPS433

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinism vs performance trade-offs
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            # cuDNN settings
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.benchmark = True
    except Exception:
        # torch not installed or not available; it's fine for CPU-only workflows
        pass


# -----------------------------
# Logging
# -----------------------------
def get_logger(
    name: str = "bert-sentiment",
    level: str | int = "INFO",
    log_file: Optional[str] = None,
    propagate: bool = False,
) -> logging.Logger:
    """
    Create a configured logger with console (and optional file) handlers.

    Args:
        name: Logger name.
        level: Logging level (str or int).
        log_file: Optional path to write logs.
        propagate: Whether to propagate to parent loggers.

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        logger.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO))
        return logger

    logger.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = propagate

    fmt = "[%(asctime)s] %(levelname)s - %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(console_handler)

    if log_file:
        # ensure directory exists
        _ensure_dir(os.path.dirname(log_file) or ".")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(file_handler)

    return logger


# -----------------------------
# Simple timer (context + decorator)
# -----------------------------
class Timer:
    """
    Usage:
        with Timer("tokenization", logger):
            do_work()

        @timeit()
        def f(...): ...
    """

    def __init__(self, name: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.name = name or "timer"
        self.logger = logger
        self.start: float = 0.0
        self.end: float = 0.0
        self.elapsed: float = math.nan

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        if self.logger:
            self.logger.info(f"[{self.name}] took {self.elapsed:.3f}s")


def timeit(name: Optional[str] = None, logger: Optional[logging.Logger] = None):
    """
    Decorator to time function execution using Timer.

    Example:
        @timeit("train_epoch", logger)
        def train_epoch(...):
            ...
    """
    def _decorator(func):
        def _wrapped(*args, **kwargs):
            _name = name or func.__name__
            with Timer(_name, logger):
                return func(*args, **kwargs)
        _wrapped.__name__ = getattr(func, "__name__", "wrapped")
        _wrapped.__doc__ = func.__doc__
        _wrapped.__dict__.update(getattr(func, "__dict__", {}))
        return _wrapped
    return _decorator


# -----------------------------
# Convenience: save/load JSON (for metrics, reports)
# -----------------------------
def save_json(obj: Dict[str, Any], path: str, indent: int = 2) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
