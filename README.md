# GlotOCR-bench

<p align="center">
<a href="https://huggingface.co/datasets/cis-lmu/GlotOCR-bench"><img alt="HuggingFace Benchmark" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Benchmark-8A2BE2"></a>
<a href="https://huggingface.co/datasets/cis-lmu/GlotOCR-bench-v1.0-results"><img alt="HuggingFace Results" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Results-blue"></a>
<a href="https://huggingface.co/datasets/uv-scripts/ocr"><img alt="HuggingFace OCR Scripts" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-OCR Scripts-orange"></a>
<a href="https://arxiv.org/abs/2604.12978"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2604.12978-b31b1b.svg"></a>
</p>

A multilingual OCR benchmark covering a wide range of writing scripts, designed to evaluate OCR models across hundreds of languages.

---

## Benchmark & Results

The benchmark dataset is available at:
[cis-lmu/GlotOCR-bench](https://huggingface.co/datasets/cis-lmu/GlotOCR-bench)

Per-model evaluation results are available at:
[cis-lmu/GlotOCR-bench-v1.0-results](https://huggingface.co/datasets/cis-lmu/GlotOCR-bench-v1.0-results)

---

## Dataset

All dataset-related code is in the `dataset/` folder.

### 1. Fonts

Download and organize Google Fonts by script by running:

```bash
python dataset/fonts/get_fonts.py
```

Alternatively, you can download the version we used directly from Hugging Face:
[kargaranamir/google_fonts](https://huggingface.co/datasets/kargaranamir/google_fonts)

### 2. Seed Text

Place per-script sentence CSVs in `dataset/seed/seed_data/`. You can generate them from the GlotLID corpus by running:

```bash
python dataset/seed/get_seed.py
```

The GlotLID corpus is available at:
[cis-lmu/glotlid-corpus](https://huggingface.co/datasets/cis-lmu/glotlid-corpus)

### 3. Image Generation

We provide two rendering profiles (`PLAIN` and `OLD_DOCUMENT`). You can adjust parameters in `dataset/ocr_generator/config.py` and the rendering logic in `dataset/ocr_generator/engine.py`, then generate images by running:

```bash
python dataset/ocr_generator/main.py
```

To export the generated images to Parquet format:

```bash
python dataset/ocr_generator/export.py
```

---

## Evaluation

### 1. Run OCR Models

Run the OCR models on the dataset using the scripts provided at:
[uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr)

### 2. Compute Metrics

Once model outputs are ready, compute CER, Acc@k, and ScriptAcc metrics by running:

```bash
cd evaluation/metrics
python main.py
```

Results are saved per model under `evaluation/res_v1.0/`, including per-script, per-language, and tier-level (high / mid / low resource) breakdowns.

---

## Citation

```bibtex
@misc{kargaran2026glotocrbench,
      title={GlotOCR Bench: OCR Models Still Struggle Beyond a Handful of Unicode Scripts}, 
      author={Amir Hossein Kargaran and Nafiseh Nikeghbal and Jana Diesner and François Yvon and Hinrich Schütze},
      year={2026},
      eprint={2604.12978},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.12978}, 
}
```