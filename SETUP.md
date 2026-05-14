# Setup & Reproducibility Guide

This repository extends [david-gimeno/interpreting-ssl-parkinson-speech](https://github.com/david-gimeno/interpreting-ssl-parkinson-speech) with support for the [Movement Disorders Voice dataset](https://www.kaggle.com/datasets/cycoool29/movement-disorders-voice) (Kaggle).

---

## 1. Environment

```bash
conda create -n ym_voice python=3.10
conda activate ym_voice
pip install -r requirements.txt
```

> Tested with CUDA 12.1, torch 2.4.1+cu121.

---

## 2. Download the Dataset

The Kaggle dataset (~4 GB) is **not** tracked in this repository. Download it manually:

### Option A — Kaggle web UI
1. Go to https://www.kaggle.com/datasets/cycoool29/movement-disorders-voice
2. Click **Download** and save as `movement-disorders-voice.zip`
3. Extract:
   ```bash
   unzip movement-disorders-voice.zip -d /path/to/movement-disorders-voice
   ```

### Option B — Kaggle API CLI
```bash
pip install kaggle          # or: conda install -c conda-forge kaggle
# Place your kaggle.json token in ~/.kaggle/kaggle.json (chmod 600)
kaggle datasets download cycoool29/movement-disorders-voice
unzip movement-disorders-voice.zip -d /path/to/movement-disorders-voice
```

> After extraction the directory should contain `data/Archive Raw Audio/Healthy a/` and `data/Archive Raw Audio/Parkinson a/`.

---

## 3. Data Preparation (Movement Disorders)

Run the full preparation pipeline from the repository root:

```bash
conda activate ym_voice
bash scripts/runs/dataset_preparation/movement_disorders.sh \
    "/path/to/movement-disorders-voice/data/Archive Raw Audio" \
    clean
```

This performs:
1. **Restructure** — renames files to GITA-compatible format and generates `data/movement_disorders/metadata.csv`
2. **Normalize** — resamples audio to 16 kHz mono via ffmpeg-normalize
3. **Split CSV** — builds `splits/movement_disorders/dataset.csv`
4. **Cross-validation** — creates 5-fold stratified splits under `splits/movement_disorders/fold_*/`

File variant options for step 1: `clean` (default), `raw`, `LF`.

> **Note on recording content.** Each file contains the patient's sustained-vowel phonation. An evaluator instruction ("say /a/…") may appear in the first ~0.5 s before phonation onset. Energy-profile analysis shows no two-speaker alternation pattern; longer files (up to 43 s) follow a single-speaker gradual-decay profile consistent with Maximum Phonation Time tasks. To be safe, you can trim the start of every file by passing `--onset-trim-sec 1.0` to the restructure script (adds a 1-second leading silence removal before normalization).

### Naming convention after restructure

| Original | Renamed |
|----------|---------|
| `HC10a1_clean.wav` | `HC_SUSTAINED-VOWELS_AVPEPUDEAC0010_A1.wav` |
| `PD10a1_clean.wav` | `PD_SUSTAINED-VOWELS_AVPEPUDEA0010_A1.wav` |

Subject mapping: `HC{N}` → `AVPEPUDEAC{N:04d}` (label 0), `PD{N}` → `AVPEPUDEA{N:04d}` (label 1).

---

## 4. Feature Extraction

```bash
bash scripts/runs/feature_extraction/movement_disorders.sh
```

Extracts:
- **DisVoice** features (articulation, glottal, phonation, prosody) → `data/movement_disorders/speech_features/disvoice/`
- **Wav2Vec2 XLSR-300M** layer-7 embeddings → `data/movement_disorders/speech_features/wav2vec/layer07/`

> Feature extraction is slow on CPU. A GPU is strongly recommended.

---

## 5. Training & Evaluation

```bash
bash scripts/runs/experiments/cross_full/movement_disorders.sh
```

Runs the `cross_full` model (cross-attention over SSL + DisVoice features) across 5 seeds × 5 folds on the `SUSTAINED-VOWELS` task.

Evaluate after training:

```bash
python scripts/evaluation/overall_performance.py \
    --exps-dir ./exps/movement_disorders/cross_full/SUSTAINED-VOWELS/
```

---

## 6. Original GITA Corpus (PC-GITA)

To reproduce the original paper experiments on the PC-GITA corpus, follow the upstream README instructions — the scripts in `scripts/runs/dataset_preparation/gita.sh`, `scripts/runs/feature_extraction/gita.sh`, and `scripts/runs/experiments/cross_full/gita.sh` remain unchanged.

---

## Dataset Summary

| Dataset | Subjects (HC / PD) | Tasks | Language |
|---------|-------------------|-------|----------|
| Movement Disorders (Kaggle) | 22 / 22 | Sustained vowels (a, i) | English |
| PC-GITA (original paper) | 50 / 50 | Vowels, DDK, sentences, read text | Spanish |

---

## Citation

If you use the original framework, please cite:

```bibtex
@article{gimeno2025unveiling,
  author={Gimeno-G{\'o}mez, David and Botelho, Catarina and Pompili, Anna and Abad, Alberto and Martínez-Hinarejos, Carlos-D.},
  title={{Unveiling Interpretability in Self-Supervised Speech Representations for Parkinson's Diagnosis}},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  year={2025},
  doi={10.1109/JSTSP.2025.3539845},
}
```
