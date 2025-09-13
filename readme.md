# Synthbuster

Implementation of [Synthbuster: Towards Detection of Diffusion Model Generated Images](https://ieeexplore.ieee.org/abstract/document/10334046).

Synthbuster is a tool to detect synthetic images based on Fourier-domain artifacts.  
It supports training models with a fixed configuration, evaluating them on datasets,  
and running inference on individual images or via a Streamlit demo.

---

## 📦 Installation

We use [Astral uv](https://docs.astral.sh/uv/) for dependency management and execution.

Clone this repository and install dependencies:

```bash
uv sync
```

---

## 📂 Dataset structure

Datasets must be organized with the following structure:

```
dataset_root/
  train/
    real/...
    fake/...
  val/
    real/...
    fake/...
  test/
    real/...
    fake/...
```

- **Synthetic test data**: download from [Zenodo](https://zenodo.org/records/10066460)  
- **Real test data**: download raise-1k from [RAISE](https://loki.disi.unitn.it/RAISE/download.html)

You can also have nested subfolders under `real/` and `fake/` (recursively scanned).

---

## 🏋️ Training a model (fixed config)

To train a model with a specific configuration:

1. Prepare a configuration file (JSON or joblib).  
   Example `config.json`:

   ```json
   {
     "dataset_root": "/path/to/dataset",
     "file_extension": ".png",
     "method": "rank",
     "rank_sz": 4,
     "max_period": 16,
     "jpeg_mode": "none",
     "learning_rate": 0.05,
     "max_iter": 200,
     "max_depth": 3,
     "max_leaf_nodes": 31,
     "min_samples_leaf": 50,
     "l2_regularization": 1e-6,
     "random_state": 753
   }
   ```

2. Train:

   ```bash
   uv run train_fixed.py \
     --config config.json \
     --save-model models/model.joblib \
     --save-config models/config.json
   ```

This will save both the trained model (`.joblib`) and the config used (`.json`).

---

## 🔍 Inference and evaluation

### 1. Single image inference

Run the trained model on a single image:

```bash
uv run infer_image.py \
  --model models/model.joblib \
  --config models/config.json \
  --image data/test/real/example.png
```

Outputs the predicted probability of being synthetic and the class decision.

---

### 2. Dataset evaluation

Evaluate the model on one of the dataset splits (`train`, `val`, `test`, or `all`):

```bash
uv run evaluate_dataset.py \
  --model models/model.joblib \
  --config models/config.json \
  --split test \
  --n-jobs -1 \
  --results results_test.json
```

Options:
- `--optimize-threshold`: re-optimize threshold for MCC on the chosen split.
- `--force-jpeg-quality Q`: compress all evaluated images at JPEG quality `Q` before evaluation.

Results are printed to stdout and saved in JSON (`results_test.json`).

---

### 3. Streamlit demo

A simple interactive demo is available:

```bash
uv run streamlit run streamlit_app.py
```

Features:
- Choose among all models in `models/*.joblib`
- Upload an image or provide an image URL
- See predictions live in the browser

---

## Citation
```bibtex
@article{synthbuster,
  author={Bammey, Quentin},
  journal={IEEE Open Journal of Signal Processing}, 
  title={Synthbuster: Towards Detection of Diffusion Model Generated Images}, 
  year={2024},
  volume={5},
  number={},
  pages={1-9},
  doi={10.1109/OJSP.2023.3337714}}
```
```
