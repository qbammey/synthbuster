# evaluate_dataset.py
"""
Way 2: Evaluate a trained model+config on a dataset split (train/val/test or all).

Examples:
  python evaluate_dataset.py --model best_model.joblib --config best_config.json \
      --split test --results results_test.json

  python evaluate_dataset.py --model best_model.joblib --config best_config.json \
      --split all --n-jobs 32 --results results_all.json

Override compression for evaluation:
  python evaluate_dataset.py --model best_model.joblib --config best_config.json \
      --split test --force-jpeg-quality 60

Behavior:
- Loads dataset_root and file_extension from the config (can override both via CLI).
- Uses preprocessing + JPEG spec exactly as saved in the config, unless --force-jpeg-quality is provided.
- Computes scores, chooses threshold (by default from config; can force optimize).
- Prints summary metrics to stdout and saves a JSON with per-image outputs + summary.
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Any, List

import numpy as np

from inference_common import (
    load_model, load_config, JPEGSpec, PreprocessConfig,
    gather_split_paths, features_for_paths, predict_proba,
    summarize_metrics
)

def _choose_threshold_from_config(cfg: dict) -> float:
    t = None
    if "test" in cfg and isinstance(cfg["test"], dict):
        t = cfg["test"].get("threshold_mcc", None)
    if t is None and "validation" in cfg and isinstance(cfg["validation"], dict):
        t = cfg["validation"].get("threshold_mcc", None)
    return float(t) if t is not None else 0.5

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Synthbuster model on a dataset split.")
    p.add_argument("--model", required=True, help="Path to trained model .joblib")
    p.add_argument("--config", required=True, help="Path to config (.json or .joblib)")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test",
                   help="Which split to evaluate (default: test). 'all' concatenates all three.")
    p.add_argument("--dataset", default=None, help="Override dataset root from config (optional).")
    p.add_argument("--ext", default=None, help="Override file extension filter (default from config).")
    p.add_argument("--n-jobs", type=int, default=0, help="Parallelism for feature extraction (threaded).")
    p.add_argument("--optimize-threshold", action="store_true",
                   help="If set, find MCC-optimal threshold on the evaluated split (ignores config threshold).")
    p.add_argument("--results", default="results.json", help="Where to save the detailed results JSON.")
    p.add_argument("--force-jpeg-quality", type=int, default=None,
                   help="If set (0..100), compress ALL evaluated images at this fixed JPEG quality, "
                        "overriding the config's JPEG settings.")
    args = p.parse_args()

    if args.force_jpeg_quality is not None and not (0 <= args.force_jpeg_quality <= 100):
        raise ValueError("--force-jpeg-quality must be between 0 and 100")

    cfg = load_config(args.config)
    model = load_model(args.model)

    dataset_root = args.dataset or cfg.get("dataset_root")
    if not dataset_root:
        raise ValueError("dataset_root missing in config and not overridden with --dataset.")
    ext = args.ext or cfg.get("file_extension", ".png")

    pre_cfg = PreprocessConfig.from_config(cfg)

    # Choose JPEG spec: forced fixed quality overrides config
    if args.force_jpeg_quality is not None:
        jpeg_spec = JPEGSpec(mode="fixed", quality=int(args.force_jpeg_quality))
        print(f"[Info] Forcing JPEG quality={args.force_jpeg_quality} for ALL images (overrides config).")
    else:
        jpeg_spec = JPEGSpec.from_config(cfg)

    # Gather paths
    splits = []
    if args.split == "all":
        for s in ("train", "val", "test"):
            paths = gather_split_paths(dataset_root, s, ext=ext)
            if paths:
                splits.append((s, paths))
    else:
        paths = gather_split_paths(dataset_root, args.split, ext=ext)
        splits.append((args.split, paths))

    # Process each split, print and accumulate results
    summary: Dict[str, Any] = {}
    all_rows: List[Dict[str, Any]] = []

    for split_name, paths in splits:
        X, y, paths_ord = features_for_paths(paths, pre_cfg, jpeg_spec, n_jobs=args.n_jobs)
        scores = predict_proba(model, X)

        if args.optimize_threshold:
            metrics, thr = summarize_metrics(y, scores, optimize_threshold=True)
            used_threshold = thr.threshold
        else:
            used_threshold = _choose_threshold_from_config(cfg)
            metrics, _ = summarize_metrics(y, scores, threshold=used_threshold, optimize_threshold=False)

        # stdout
        print(f"\n=== Split: {split_name} ===")
        print(f"Images   : {len(paths_ord)}")
        print(f"Threshold: {used_threshold:.6f}")
        print(f"MCC      : {metrics['mcc']:.6f}")
        print(f"Accuracy : {metrics['acc']:.6f}")
        print(f"F1-score : {metrics['f1']:.6f}")
        print(f"Precision: {metrics['precision']:.6f}")
        print(f"Recall   : {metrics['recall']:.6f}")
        print(f"ROC AUC  : {metrics['auc'] if metrics['auc'] is not None else float('nan'):.6f}")
        print(f"Confusion: {metrics['confusion_matrix']}")

        # detailed rows
        preds = (scores >= used_threshold).astype(int)
        for pth, prob, lbl, pred in zip(paths_ord, scores.tolist(), y.tolist(), preds.tolist()):
            all_rows.append({
                "split": split_name,
                "path": pth,
                "prob_fake": float(prob),
                "label": int(lbl),
                "pred": int(pred),
            })

        # summary per split
        summary[split_name] = {
            "count": len(paths_ord),
            "threshold": float(used_threshold),
            **metrics,
        }

    # Save results JSON
    out = {
        "model_path": os.path.abspath(args.model),
        "config_path": os.path.abspath(args.config),
        "dataset_root": os.path.abspath(dataset_root),
        "file_extension": ext,
        "preprocess": {
            "method": pre_cfg.method,
            "rank_sz": pre_cfg.rank_sz,
            "max_period": pre_cfg.max_period,
        },
        "jpeg_spec_used": {  # record the effective spec used during this evaluation
            "mode": jpeg_spec.mode,
            "quality": jpeg_spec.quality,
            "qmin": jpeg_spec.qmin,
            "qmax": jpeg_spec.qmax,
            "seed": jpeg_spec.seed,
        },
        "summary": summary,
        "predictions": all_rows,
    }
    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[+] Saved detailed results to {args.results}")

if __name__ == "__main__":
    main()

