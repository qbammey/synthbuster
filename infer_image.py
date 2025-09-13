# infer_image.py
"""
Way 1: Inference on a single image.

Usage:
  python infer_image.py --model models/best_model.joblib --config configs/best_config.json \
                        --image /path/to/img.png

Prints:
  - P(fake)
  - Predicted label (fake=1 / real=0)
  - Threshold used (defaults to: prefer 'test.threshold_mcc' in config; else 'validation.threshold_mcc'; else 0.5)
"""

from __future__ import annotations
import argparse
import os
import sys

from inference_common import (
    load_model, load_config, JPEGSpec, PreprocessConfig,
    load_image_any, image_to_features,
)

def _choose_threshold_from_config(cfg: dict) -> float:
    # Prefer test threshold if available, else validation, else 0.5
    t = None
    if "test" in cfg and isinstance(cfg["test"], dict):
        t = cfg["test"].get("threshold_mcc", None)
    if t is None and "validation" in cfg and isinstance(cfg["validation"], dict):
        t = cfg["validation"].get("threshold_mcc", None)
    return float(t) if t is not None else 0.5

def main() -> None:
    p = argparse.ArgumentParser(description="Synthbuster single-image inference.")
    p.add_argument("--model", required=True, help="Path to trained model .joblib")
    p.add_argument("--config", required=True, help="Path to config (.json or .joblib)")
    p.add_argument("--image", required=True, help="Path or URL to image")
    args = p.parse_args()

    cfg = load_config(args.config)
    model = load_model(args.model)

    pre_cfg = PreprocessConfig.from_config(cfg)
    jpeg_spec = JPEGSpec.from_config(cfg)

    img = load_image_any(args.image)
    q = jpeg_spec.quality_for_path(args.image)  # range/fixed/none
    feat = image_to_features(img, pre_cfg, jpeg_q=q).reshape(1, -1)

    # Probability of fake
    proba_fake = model.predict_proba(feat)[0, 1]
    threshold = _choose_threshold_from_config(cfg)
    label = int(proba_fake >= threshold)

    print(f"Image: {args.image}")
    print(f"P(fake): {proba_fake:.6f}")
    print(f"Threshold used: {threshold:.6f}")
    print(f"Predicted label: {label}  (1=fake, 0=real)")

if __name__ == "__main__":
    main()

