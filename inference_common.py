# inference_common.py
"""
Shared inference utilities for Synthbuster.

This module centralizes:
- Loading model (.joblib) and config (.json or .joblib)
- JPEG spec (none/fixed/range) handling
- Preprocessing → FFT feature extraction (consistent with training)
- Threshold selection (MCC w/ margin tie-break) and decision
- Dataset discovery and batch evaluation
- Metrics computation and results packaging

All consumers (CLI scripts + Streamlit) import from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any

import glob
import hashlib
import io
import json
import math
import os
import sys

import joblib
import numpy as np
import imageio.v3 as iio
import requests
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --- Make local imports work regardless of CWD ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from preprocess import preprocess_for_fft_features  # noqa: E402
from utils import jpeg_compress  # noqa: E402


# ----------------------------
# Config & model I/O
# ----------------------------

def load_config(path: str) -> Dict[str, Any]:
    """
    Load a config saved by the training scripts.
    Accepts .json (preferred) or .joblib containing a plain dict.

    Returns
    -------
    dict
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("JSON config did not contain a dict.")
        return cfg
    # fallback: joblib
    cfg = joblib.load(path)
    if not isinstance(cfg, dict):
        raise ValueError("Joblib config did not contain a dict.")
    return cfg


def load_model(path: str) -> HistGradientBoostingClassifier:
    """
    Load a scikit-learn model saved with joblib.
    """
    model = joblib.load(path)
    if not isinstance(model, HistGradientBoostingClassifier):
        # Still allow other sklearn classifiers if needed, but warn.
        try:
            _ = model.predict_proba  # type: ignore[attr-defined]
        except Exception as e:
            raise TypeError(f"Loaded object is not a compatible classifier: {type(model)}") from e
    return model  # type: ignore[return-value]


# ----------------------------
# JPEG compression spec (fixed / range / none)
# ----------------------------

@dataclass(frozen=True)
class JPEGSpec:
    mode: str  # 'none', 'fixed', 'range'
    quality: Optional[int] = None
    qmin: Optional[int] = None
    qmax: Optional[int] = None
    seed: Optional[int] = None

    def key(self) -> Tuple:
        if self.mode == "none":
            return ("none",)
        if self.mode == "fixed":
            return ("fixed", int(self.quality))
        return ("range", int(self.qmin), int(self.qmax), int(self.seed))

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "JPEGSpec":
        d = cfg.get("jpeg_spec", {"mode": "none"})
        return JPEGSpec(
            mode=d.get("mode", "none"),
            quality=d.get("quality"),
            qmin=d.get("qmin"),
            qmax=d.get("qmax"),
            seed=d.get("seed"),
        )

    def quality_for_path(self, path: str) -> Optional[int]:
        if self.mode == "none":
            return None
        if self.mode == "fixed":
            return int(self.quality) if self.quality is not None else None
        # deterministic per-path in [qmin, qmax]
        assert self.qmin is not None and self.qmax is not None
        seed = 1234 if self.seed is None else int(self.seed)
        lo, hi = int(self.qmin), int(self.qmax)
        if lo > hi:
            lo, hi = hi, lo
        h = hashlib.sha256()
        h.update(path.encode("utf-8"))
        h.update(str(seed).encode("utf-8"))
        val = int.from_bytes(h.digest()[:8], "little", signed=False)
        span = hi - lo + 1
        return lo + (val % span)


# ----------------------------
# Image I/O (path or URL)
# ----------------------------

def load_image_any(source: str) -> np.ndarray:
    """
    Load an image from a local path or an HTTP(S) URL.
    Returns a uint8 array with shape (Y, X) or (Y, X, 3). Drops alpha if present.
    """
    if source.startswith("http://") or source.startswith("https://"):
        resp = requests.get(source, timeout=15)
        resp.raise_for_status()
        data = io.BytesIO(resp.content)
        img = iio.imread(data)
    else:
        img = iio.imread(source)

    if img.ndim == 2:
        pass
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        if img.shape[2] == 4:
            img = img[..., :3]
    else:
        raise ValueError(f"Unsupported image shape {img.shape} from {source}")

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8, copy=False)
    return img


# ----------------------------
# Preprocessing → features
# ----------------------------

@dataclass(frozen=True)
class PreprocessConfig:
    method: str
    rank_sz: int
    max_period: int

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "PreprocessConfig":
        pre = cfg["preprocess"]
        return PreprocessConfig(
            method=pre["method"],
            rank_sz=int(pre["rank_sz"]),
            max_period=int(pre["max_period"]),
        )


def image_to_features(
    img: np.ndarray,
    pre_cfg: PreprocessConfig,
    jpeg_q: Optional[int] = None,
) -> np.ndarray:
    """
    Optionally JPEG-compress, then extract Synthbuster FFT features.
    """
    if jpeg_q is not None:
        img = jpeg_compress(img, int(jpeg_q))
    feat = preprocess_for_fft_features(
        img, method=pre_cfg.method, rank_sz=pre_cfg.rank_sz, max_period=pre_cfg.max_period
    )
    # Model was trained on float32 features
    return feat.astype(np.float32, copy=False)


# ----------------------------
# Threshold search (MCC with margin tie-break)
# ----------------------------

@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    mcc: float
    margin: float

def best_threshold_mcc_with_margin(scores: np.ndarray, y_true: np.ndarray) -> ThresholdResult:
    """
    Find threshold t maximizing MCC(y_true, scores >= t).
    Tie-break equal MCC by largest separation margin.

    Returns
    -------
    ThresholdResult(threshold, mcc, margin)
    """
    if scores.size != y_true.size:
        raise ValueError("scores and y_true must have same length")
    n = scores.size
    if n == 0:
        return ThresholdResult(0.5, 0.0, 0.0)

    s = np.clip(scores.astype(np.float64, copy=False), 0.0, 1.0)
    y = y_true.astype(np.int64, copy=False)

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    pos_total = int((y == 1).sum())
    neg_total = n - pos_total
    pos_prefix = np.cumsum(y_sorted == 1)
    neg_prefix = np.cumsum(y_sorted == 0)

    uniq = np.unique(s_sorted)
    if uniq.size == 1:
        cand = [max(0.0, uniq[0] - 1e-6), min(1.0, uniq[0] + 1e-6)]
    else:
        mids = 0.5 * (uniq[:-1] + uniq[1:])
        cand = [max(0.0, uniq[0] - 1e-9)] + list(mids) + [min(1.0, uniq[-1] + 1e-9)]

    last_neg_score = np.full(n, np.nan)
    last_seen_neg = np.nan
    for i in range(n):
        if y_sorted[i] == 0:
            last_seen_neg = s_sorted[i]
        last_neg_score[i] = last_seen_neg

    first_pos_score_from_right = np.full(n, np.nan)
    next_seen_pos = np.nan
    for i in range(n - 1, -1, -1):
        if y_sorted[i] == 1:
            next_seen_pos = s_sorted[i]
        first_pos_score_from_right[i] = next_seen_pos

    best_mcc = -1.0
    best_t = 0.5
    best_margin = -1.0

    for t in cand:
        k = np.searchsorted(s_sorted, t, side="left")
        tn = int(neg_prefix[k - 1]) if k > 0 else 0
        fn = int(pos_prefix[k - 1]) if k > 0 else 0
        fp = neg_total - tn
        tp = pos_total - fn

        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = 0.0 if denom == 0.0 else ((tp * tn - fp * fn) / denom)

        left_neg = last_neg_score[k - 1] if k > 0 else np.nan
        right_pos = first_pos_score_from_right[k] if k < n else np.nan
        left_gap = (t - left_neg) if not np.isnan(left_neg) else 0.0
        right_gap = (right_pos - t) if not np.isnan(right_pos) else 0.0
        margin = min(left_gap, right_gap)

        if (mcc > best_mcc) or (abs(mcc - best_mcc) <= 1e-12 and margin > best_margin):
            best_mcc = mcc
            best_t = float(t)
            best_margin = float(max(margin, 0.0))

    return ThresholdResult(threshold=best_t, mcc=float(best_mcc), margin=float(best_margin))


# ----------------------------
# Dataset discovery and labels
# ----------------------------

def _sorted_existing_files(pattern: str) -> List[str]:
    files = glob.glob(pattern, recursive=True)
    files.sort()
    return files

def _infer_binary_label_from_path(path: str) -> int:
    parts_lower = [p.lower() for p in os.path.normpath(path).split(os.sep)]
    has_real = any("real" in p for p in parts_lower)
    has_fake = any("fake" in p for p in parts_lower)
    if has_real and not has_fake:
        return 0
    if has_fake and not has_real:
        return 1
    raise ValueError(f"Could not infer label from '{path}' (need parent 'real' or 'fake').")

def gather_split_paths(root: str, split: str, ext: str = ".png") -> List[str]:
    split_dir = os.path.join(root, split)
    real_dir = os.path.join(split_dir, "real")
    fake_dir = os.path.join(split_dir, "fake")
    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        raise FileNotFoundError(
            f"Expected '{split}/real' and '{split}/fake' under '{root}'."
        )
    ext_no_dot = ext[1:] if ext.startswith(".") else ext
    patterns = [
        os.path.join(real_dir, f"**/*.{ext_no_dot}"),
        os.path.join(real_dir, f"**/*.{ext_no_dot.upper()}"),
        os.path.join(fake_dir, f"**/*.{ext_no_dot}"),
        os.path.join(fake_dir, f"**/*.{ext_no_dot.upper()}"),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(_sorted_existing_files(pat))
    files = [p for p in files if os.path.isfile(p)]
    if not files:
        raise FileNotFoundError(f"No '*.{ext_no_dot}' found under '{real_dir}' or '{fake_dir}'.")
    return files


# ----------------------------
# Batch feature extraction & prediction
# ----------------------------

def features_for_paths(
    paths: Sequence[str],
    pre_cfg: PreprocessConfig,
    jpeg_spec: JPEGSpec,
    n_jobs: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract features for a list of paths (optionally threaded).
    Returns (X, y, paths_sorted) aligned.
    """
    def _one(p: str) -> Tuple[np.ndarray, int, str]:
        img = load_image_any(p)
        q = jpeg_spec.quality_for_path(p)
        feat = image_to_features(img, pre_cfg, jpeg_q=q)
        lbl = _infer_binary_label_from_path(p)
        return feat, lbl, p

    if n_jobs in (0, 1):
        results = [_one(p) for p in paths]
    else:
        results = Parallel(
            n_jobs=n_jobs, backend="threading", verbose=0, batch_size=1
        )(delayed(_one)(p) for p in paths)

    X = np.vstack([r[0] for r in results]).astype(np.float32, copy=False)
    y = np.asarray([r[1] for r in results], dtype=np.int64)
    ordered = [r[2] for r in results]
    return X, y, ordered


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Return P(fake) for each row in X using model.predict_proba.
    """
    probs = model.predict_proba(X)  # (N, 2)
    return probs[:, 1]


def summarize_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
    optimize_threshold: bool = False,
) -> Tuple[Dict[str, Any], ThresholdResult]:
    """
    Compute metrics at MCC-optimized threshold (or at provided threshold).
    Returns (metrics_dict, threshold_result).
    """
    if optimize_threshold or threshold is None:
        thr = best_threshold_mcc_with_margin(scores, y_true)
    else:
        thr = ThresholdResult(threshold=float(threshold), mcc=0.0, margin=0.0)
        # compute MCC at provided threshold
        pred = (scores >= thr.threshold).astype(int)
        thr = ThresholdResult(
            threshold=thr.threshold,
            mcc=float(matthews_corrcoef(y_true, pred)),
            margin=0.0,  # unknown; only meaningful for optimized threshold
        )

    pred = (scores >= thr.threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")

    metrics = {
        "threshold": float(thr.threshold),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "acc": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "auc": float(auc) if not math.isnan(auc) else None,
        "margin": float(thr.margin),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }
    return metrics, thr

