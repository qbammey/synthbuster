# app_streamlit.py
"""
Way 3: Streamlit demo for Synthbuster.

Features:
- Pick any model from models/*.joblib (auto-discovers on launch).
- Pick a matching config file (auto-guess from common names; or browse).
- Upload an image or paste an URL.
- Shows P(fake), threshold used (from config), and predicted label.
- Displays the image.

Run:
  streamlit run app_streamlit.py
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import numpy as np
import streamlit as st

from inference_common import (
    load_model, load_config, JPEGSpec, PreprocessConfig,
    load_image_any, image_to_features
)

MODELS_DIR = "models"

def _list_models() -> List[str]:
    return sorted(glob.glob(os.path.join(MODELS_DIR, "*.joblib")))

def _choose_threshold_from_config(cfg: dict) -> float:
    t = None
    if "test" in cfg and isinstance(cfg["test"], dict):
        t = cfg["test"].get("threshold_mcc", None)
    if t is None and "validation" in cfg and isinstance(cfg["validation"], dict):
        t = cfg["validation"].get("threshold_mcc", None)
    return float(t) if t is not None else 0.5

def _guess_config_for_model(model_path: str) -> Optional[str]:
    """
    Try to find a config next to the model or in current directory with similar stem.
    """
    stem = os.path.splitext(os.path.basename(model_path))[0]
    candidates = [
        os.path.join(os.path.dirname(model_path), stem + ".json"),
        os.path.join(os.path.dirname(model_path), stem.replace("model", "config") + ".json"),
        stem + ".json",
        stem.replace("model", "config") + ".json",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # also accept joblib configs
    candidates = [c[:-5] + ".joblib" for c in candidates if c.endswith(".json")]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

def main() -> None:
    st.set_page_config(page_title="Synthbuster Demo", page_icon="🕵️", layout="centered")

    st.title("🕵️ Synthbuster – Synthetic Image Detector")

    models = _list_models()
    if not models:
        st.warning(f"No models found in `{MODELS_DIR}/*.joblib`.")
        return

    model_path = st.selectbox("Choose a model", models, index=0, help="Models discovered in models/*.joblib")
    default_cfg = _guess_config_for_model(model_path)
    cfg_path = st.text_input("Config path (.json or .joblib)", value=default_cfg or "", help="Provide the matching config for the selected model")

    load_clicked = st.button("Load model & config")
    if "state_loaded" not in st.session_state:
        st.session_state.state_loaded = False

    if load_clicked:
        try:
            cfg = load_config(cfg_path)
            model = load_model(model_path)
            pre_cfg = PreprocessConfig.from_config(cfg)
            jpeg_spec = JPEGSpec.from_config(cfg)
            threshold = _choose_threshold_from_config(cfg)

            st.session_state.state_loaded = True
            st.session_state.cfg = cfg
            st.session_state.model = model
            st.session_state.pre_cfg = pre_cfg
            st.session_state.jpeg_spec = jpeg_spec
            st.session_state.threshold = threshold

            st.success("Loaded model & config successfully.")
        except Exception as e:
            st.session_state.state_loaded = False
            st.error(f"Failed to load: {e}")

    if not st.session_state.state_loaded:
        st.info("Load a model & config to begin.")
        return

    st.subheader("Predict")
    tab1, tab2 = st.tabs(["Upload image", "From URL"])

    with tab1:
        up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])
        if up is not None:
            import imageio.v3 as iio
            data = up.read()
            try:
                import io
                img = iio.imread(io.BytesIO(data))
                if img.ndim == 3 and img.shape[2] == 4:
                    img = img[..., :3]
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8, copy=False)

                q = st.session_state.jpeg_spec.quality_for_path(up.name)  # deterministic by name
                feat = image_to_features(img, st.session_state.pre_cfg, jpeg_q=q).reshape(1, -1)
                p_fake = st.session_state.model.predict_proba(feat)[0, 1]
                label = int(p_fake >= st.session_state.threshold)

                st.image(img, caption=f"Uploaded: {up.name}", use_column_width=True)
                st.metric("P(fake)", f"{p_fake:.6f}")
                st.write(f"Threshold: **{st.session_state.threshold:.6f}**  →  Predicted label: **{label}** (1=fake, 0=real)")
            except Exception as e:
                st.error(f"Failed to read/process image: {e}")

    with tab2:
        url = st.text_input("Image URL (http/https)")
        if st.button("Predict from URL", type="primary") and url:
            try:
                img = load_image_any(url)
                q = st.session_state.jpeg_spec.quality_for_path(url)
                feat = image_to_features(img, st.session_state.pre_cfg, jpeg_q=q).reshape(1, -1)
                p_fake = st.session_state.model.predict_proba(feat)[0, 1]
                label = int(p_fake >= st.session_state.threshold)

                st.image(img, caption=url, use_column_width=True)
                st.metric("P(fake)", f"{p_fake:.6f}")
                st.write(f"Threshold: **{st.session_state.threshold:.6f}**  →  Predicted label: **{label}** (1=fake, 0=real)")
            except Exception as e:
                st.error(f"Failed to fetch/process image: {e}")

    st.caption("Model picks threshold from config (prefers test threshold, then validation; falls back to 0.5 if absent).")

if __name__ == "__main__":
    main()

