# train_from_config.py
"""
Refit Synthbuster model from a fixed configuration (no Optuna search).

Inputs
------
--config: path to a .joblib config (as produced by the search script, but it may
          omit metrics). If a JSON is provided by mistake, it will be accepted.

The config must at least contain:
{
  "dataset_root": <str>,              # can be overridden via --dataset
  "file_extension": ".png",           # optional; default ".png" if missing
  "jpeg_spec": {                      # exactly as saved by the search script
      "mode": "none" | "fixed" | "range",
      "quality": <int or null>,
      "qmin": <int or null>,
      "qmax": <int or null>,
      "seed": <int or null>
  },
  "preprocess": {
      "method": "cross" | "rank",
      "rank_sz": <int>,
      "max_period": 8 | 16
  },
  "hgb_params": { ... sklearn params ... }
}

Behavior
--------
- Loads dataset (train/val/test) recursively from dataset_root (or --dataset).
- Extracts features with the fixed preprocessing & JPEG spec.
- Computes MCC-optimal threshold on validation (for info).
- Fits final model on (train+val) and evaluates on test at its own MCC-optimal threshold.
- Saves:
    * Model           -> --save-model        (default: refit_model.joblib)
    * Updated config  -> --save-config       (default: refit_config.json)
      (includes chosen validation & test thresholds and metrics)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from collections import defaultdict

import imageio.v3 as iio
import joblib
import numpy as np
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

from preprocess import preprocess_for_fft_features
from utils import jpeg_compress


# ----------------------------
# Dataset discovery (recursive)
# ----------------------------

def _sorted_existing_files(pattern: str) -> List[str]:
    files = glob.glob(pattern, recursive=True)
    files.sort()
    return files


def _gather_split(root: str, split: str, ext: str) -> List[str]:
    split_dir = os.path.join(root, split)
    real_dir = os.path.join(split_dir, "real")
    fake_dir = os.path.join(split_dir, "fake")

    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        raise FileNotFoundError(
            f"Expected '{split}/real' and '{split}/fake' under dataset root '{root}'. "
            f"Missing:{' ' + real_dir if not os.path.isdir(real_dir) else ''}"
            f"{' ' + fake_dir if not os.path.isdir(fake_dir) else ''}"
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
        raise FileNotFoundError(
            f"No files with extension '{ext}' found recursively under '{real_dir}' or '{fake_dir}'."
        )
    files.sort()
    return files


def _gather_dataset_paths(dataset_root: str, ext: str = ".png") -> Tuple[List[str], List[str], List[str]]:
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    train_paths = _gather_split(dataset_root, "train", ext)
    val_paths = _gather_split(dataset_root, "val", ext)
    test_paths = _gather_split(dataset_root, "test", ext)
    return train_paths, val_paths, test_paths


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

    def quality_for_path(self, path: str) -> Optional[int]:
        if self.mode == "none":
            return None
        if self.mode == "fixed":
            return int(self.quality)
        # deterministic per-path in [qmin, qmax]
        assert self.qmin is not None and self.qmax is not None and self.seed is not None
        lo, hi = int(self.qmin), int(self.qmax)
        if lo > hi:
            lo, hi = hi, lo
        h = hashlib.sha256()
        h.update(path.encode("utf-8"))
        h.update(str(self.seed).encode("utf-8"))
        val = int.from_bytes(h.digest()[:8], "little", signed=False)
        span = hi - lo + 1
        return lo + (val % span)


# ----------------------------
# Data I/O & features
# ----------------------------

def _infer_binary_label_from_path(path: str) -> int:
    parts_lower = [p.lower() for p in os.path.normpath(path).split(os.sep)]
    has_real = any("real" in p for p in parts_lower)
    has_fake = any("fake" in p for p in parts_lower)
    if has_real and not has_fake:
        return 0
    if has_fake and not has_real:
        return 1
    raise ValueError(f"Could not infer label from '{path}' (need parent folder 'real' or 'fake').")


def _read_image(path: str) -> np.ndarray:
    img = iio.imread(path)
    if img.ndim == 2:
        pass
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        if img.shape[2] == 4:
            img = img[..., :3]
    else:
        raise ValueError(f"Unsupported shape {img.shape} for '{path}'.")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8, copy=False)
    return img


@dataclass(frozen=True)
class PreprocessConfig:
    method: str
    rank_sz: int
    max_period: int

    def key(self) -> Tuple[str, int, int]:
        return (self.method, int(self.rank_sz), int(self.max_period))


def _extract_feature_for_path(
    path: str,
    cfg_pre: PreprocessConfig,
    jpeg_spec: JPEGSpec,
) -> Tuple[np.ndarray, int]:
    lbl = _infer_binary_label_from_path(path)
    img = _read_image(path)
    q = jpeg_spec.quality_for_path(path)
    if q is not None:
        img = jpeg_compress(img, q)
    feat = preprocess_for_fft_features(img, method=cfg_pre.method, rank_sz=cfg_pre.rank_sz, max_period=cfg_pre.max_period)
    return feat, lbl


def _extract_features(
    paths: Sequence[str],
    cfg_pre: PreprocessConfig,
    jpeg_spec: JPEGSpec,
    n_jobs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_jobs in (0, 1):
        results = [_extract_feature_for_path(p, cfg_pre, jpeg_spec) for p in paths]
    else:
        results = Parallel(
            n_jobs=n_jobs,
            backend="threading",
            verbose=0,
            batch_size=1,
        )(delayed(_extract_feature_for_path)(p, cfg_pre, jpeg_spec) for p in paths)
    X = np.vstack([r[0] for r in results]).astype(np.float32, copy=False)
    y = np.asarray([r[1] for r in results], dtype=np.int64)
    return X, y


def _build_estimator(params: Dict[str, object]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(params["learning_rate"]),
        max_iter=int(params["max_iter"]),
        max_depth=None if params.get("max_depth") is None else int(params["max_depth"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        early_stopping=False,
        random_state=int(params["random_state"]),
        categorical_features=None,
        class_weight=None,
        scoring=None,
    )


# ----------------------------
# Threshold search (MCC with margin tie-break)
# ----------------------------

@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    mcc: float
    margin: float


def _best_threshold_mcc_with_margin(scores: np.ndarray, y_true: np.ndarray) -> ThresholdResult:
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
# Main refit flow
# ----------------------------

def _load_config_any(path: str) -> Dict[str, object]:
    """
    Load config from .joblib (preferred). If that fails, try JSON.
    """
    try:
        cfg = joblib.load(path)
        if not isinstance(cfg, dict):
            raise ValueError("Joblib file did not contain a dict.")
        return cfg
    except Exception:
        # Fallback: JSON
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("JSON file did not contain a dict.")
        return cfg


def train_from_config(
    config_path: str,
    dataset_root_override: Optional[str],
    n_jobs_extract: int,
    save_model: str,
    save_config: str,
) -> None:
    # Load config (joblib preferred; JSON accepted)
    cfg_in = _load_config_any(config_path)

    # Extract required sections
    dataset_root = dataset_root_override or cfg_in.get("dataset_root")
    if not dataset_root:
        raise ValueError("dataset_root missing from config and no --dataset provided.")

    ext = cfg_in.get("file_extension", ".png")

    jpeg_dict = cfg_in.get("jpeg_spec", {"mode": "none"})
    jpeg_spec = JPEGSpec(
        mode=jpeg_dict.get("mode", "none"),
        quality=jpeg_dict.get("quality"),
        qmin=jpeg_dict.get("qmin"),
        qmax=jpeg_dict.get("qmax"),
        seed=jpeg_dict.get("seed"),
    )

    pre = cfg_in["preprocess"]
    pre_cfg = PreprocessConfig(
        method=pre["method"],
        rank_sz=int(pre["rank_sz"]),
        max_period=int(pre["max_period"]),
    )

    hgb_params = cfg_in["hgb_params"]

    # Discover dataset
    train_paths, val_paths, test_paths = _gather_dataset_paths(dataset_root, ext=ext)

    # Preflight
    print(f"[Preflight] train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")
    if train_paths:
        feat0, lbl0 = _extract_feature_for_path(train_paths[0], pre_cfg, jpeg_spec)
        print(f"[Preflight] example: {train_paths[0]} OK (feature length={len(feat0)}, label={lbl0})")

    # Feature extraction
    X_tr, y_tr = _extract_features(train_paths, pre_cfg, jpeg_spec, n_jobs=n_jobs_extract)
    X_va, y_va = _extract_features(val_paths, pre_cfg, jpeg_spec, n_jobs=n_jobs_extract)
    X_te, y_te = _extract_features(test_paths, pre_cfg, jpeg_spec, n_jobs=n_jobs_extract)

    # Validation threshold (for reference)
    clf = _build_estimator(hgb_params)
    clf.fit(X_tr, y_tr)
    proba_va = clf.predict_proba(X_va)[:, 1]
    thr_va = _best_threshold_mcc_with_margin(proba_va, y_va)
    pred_va = (proba_va >= thr_va.threshold).astype(int)
    val_metrics = {
        "threshold_mcc": float(thr_va.threshold),
        "mcc": float(thr_va.mcc),
        "margin": float(thr_va.margin),
        "acc": float(accuracy_score(y_va, pred_va)),
        "f1": float(f1_score(y_va, pred_va, zero_division=0)),
        "precision": float(precision_score(y_va, pred_va, zero_division=0)),
        "recall": float(recall_score(y_va, pred_va, zero_division=0)),
        "auc": float(roc_auc_score(y_va, proba_va)) if len(np.unique(y_va)) > 1 else None,
    }

    # Final fit on train+val
    X_trv = np.vstack([X_tr, X_va]).astype(np.float32, copy=False)
    y_trv = np.concatenate([y_tr, y_va]).astype(np.int64, copy=False)
    final_clf = _build_estimator(hgb_params)
    final_clf.fit(X_trv, y_trv)

    # Test metrics at MCC-optimal threshold
    proba_te = final_clf.predict_proba(X_te)[:, 1]
    thr_te = _best_threshold_mcc_with_margin(proba_te, y_te)
    pred_te = (proba_te >= thr_te.threshold).astype(int)

    try:
        auc_te = roc_auc_score(y_te, proba_te)
    except ValueError:
        auc_te = float("nan")

    acc_te = accuracy_score(y_te, pred_te)
    f1_te = f1_score(y_te, pred_te, zero_division=0)
    mcc_te = matthews_corrcoef(y_te, pred_te)
    prec_te = precision_score(y_te, pred_te, zero_division=0)
    rec_te = recall_score(y_te, pred_te, zero_division=0)
    cm_te = confusion_matrix(y_te, pred_te)

    print("\n=== Validation (fixed config, MCC-optimized) ===")
    print(f"Threshold : {thr_va.threshold:.6f} (margin {thr_va.margin:.6g})")
    print(f"MCC       : {thr_va.mcc:.6f}")
    print(f"Accuracy  : {val_metrics['acc']:.6f}")
    print(f"F1-score  : {val_metrics['f1']:.6f}")
    print(f"Precision : {val_metrics['precision']:.6f}")
    print(f"Recall    : {val_metrics['recall']:.6f}")
    print(f"ROC AUC   : {val_metrics['auc'] if val_metrics['auc'] is not None else float('nan'):.6f}")

    print("\n=== Test (final model, MCC-optimized) ===")
    print(f"Threshold : {thr_te.threshold:.6f} (margin {thr_te.margin:.6g})")
    print(f"MCC       : {mcc_te:.6f}")
    print(f"Accuracy  : {acc_te:.6f}")
    print(f"F1-score  : {f1_te:.6f}")
    print(f"Precision : {prec_te:.6f}")
    print(f"Recall    : {rec_te:.6f}")
    print(f"ROC AUC   : {auc_te if not math.isnan(auc_te) else float('nan'):.6f}")
    print("Confusion matrix (rows: true [0,1], cols: pred [0,1]):")
    print(cm_te)

    # Save model
    joblib.dump(final_clf, save_model)
    print(f"\n[+] Saved model to {save_model}")

    # Save updated config (JSON) with thresholds and metrics
    out_cfg = {
        "dataset_root": os.path.abspath(dataset_root),
        "file_extension": ext,
        "jpeg_spec": {
            "mode": jpeg_spec.mode,
            "quality": jpeg_spec.quality,
            "qmin": jpeg_spec.qmin,
            "qmax": jpeg_spec.qmax,
            "seed": jpeg_spec.seed,
        },
        "preprocess": {
            "method": pre_cfg.method,
            "rank_sz": pre_cfg.rank_sz,
            "max_period": pre_cfg.max_period,
        },
        "hgb_params": hgb_params,
        "validation": val_metrics,
        "test": {
            "threshold_mcc": float(thr_te.threshold),
            "mcc": float(mcc_te),
            "margin": float(thr_te.margin),
            "acc": float(acc_te),
            "f1": float(f1_te),
            "precision": float(prec_te),
            "recall": float(rec_te),
            "auc": float(auc_te) if not math.isnan(auc_te) else None,
            "confusion_matrix": cm_te.tolist(),
        },
        "counts": {"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)},
    }
    with open(save_config, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2)
    print(f"[+] Saved refit config+metrics to {save_config}")


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Refit Synthbuster from a fixed config (no Optuna).")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config .joblib (preferred). JSON is also accepted.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset root from config (optional).")
    parser.add_argument("--n-jobs-extract", type=int, default=0,
                        help="Parallel jobs for feature extraction. Use -1 for all cores (threaded).")
    parser.add_argument("--save-model", type=str, default="refit_model.joblib",
                        help="Path to save the fitted model.")
    parser.add_argument("--save-config", type=str, default="refit_config.json",
                        help="Path to save the updated config with thresholds/metrics (JSON).")
    args = parser.parse_args()

    train_from_config(
        config_path=args.config,
        dataset_root_override=args.dataset,
        n_jobs_extract=args.n_jobs_extract,
        save_model=args.save_model,
        save_config=args.save_config,
    )


if __name__ == "__main__":
    main()

