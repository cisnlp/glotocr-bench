import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR  = "../res_v1.0"
OUTPUT_NAME = "ocr_boxplot_latn"

MODEL_NAMES = {
    "dots-ocr-1.5+img_plain":                  "dots.ocr-1.5",
    "paddleocr-vl-1.5+img_plain":              "PaddleOCR-VL-1.5",
    "gemini-3.1-flash-lite-preview+img_plain":  "Gemini 3.1 Flash-Lite",
    "gpt-4.1+img_plain":                        "GPT4.1",
    "dots-ocr+img_plain":                       "dots.ocr",
    "glm-ocr-v2+img_plain":                     "GLM-OCR",
    "deepseek-ocr2-vllm+img_plain":             "DeepSeek-OCR-2",
    "olmocr2-vllm+img_plain":                   "olmOCR-2",
    "nanonets-ocr2+img_plain":                  "Nanonets-OCR2",
    "firered-ocr+img_plain":                    "FireRed-OCR",
    "lighton-ocr2+img_plain":                   "LightOnOCR-2",
    "rolm-ocr+img_plain":                       "RolmOCR",
    "hunyuan-ocr+img_plain":                    "HunyuanOCR",
    "qwen3-vl-8b+img_plain":                    "Qwen3-VL-8B",
}

COLORS = [
    "#E07A5F",
    "#4D908E",
    "#E09F3E",
    "#577590",
    "#8E7DBE",
    "#43AA8B",
    "#F94144",
    "#277DA1",
    "#F4A261",
    "#90BE6D",
    "#C77DFF",
    "#E76F51",
    "#2A9D8F",
    "#E9C46A",
]


def load_lang_metrics(model_dir):
    path = os.path.join(model_dir, "language_metrics_Latn.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def main():
    models = sorted([
        d for d in os.listdir(RESULT_DIR)
        if os.path.isdir(os.path.join(RESULT_DIR, d))
    ])

    model_data = []
    for model in models:
        data = load_lang_metrics(os.path.join(RESULT_DIR, model))
        if data is None:
            continue
        scores = [v["Acc@0.05"] * 100 for v in data.values() if "Acc@0.05" in v]
        if not scores:
            continue
        nice = MODEL_NAMES.get(model, model)
        print(
            f"{nice:25s}  n={len(scores):3d}  "
            f"min={min(scores):.1f}  "
            f"q1={np.percentile(scores, 25):.1f}  "
            f"median={np.median(scores):.1f}  "
            f"q3={np.percentile(scores, 75):.1f}  "
            f"max={max(scores):.1f}"
        )
        model_data.append((model, scores))

    # sort by median descending
    model_data.sort(key=lambda x: np.median(x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(max(10, len(model_data) * 1.1), 6))
    # background = "#f6f6f6"
    background = "white"  # or "#ffffff"
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    positions = list(range(len(model_data)))

    for i, (model, scores) in enumerate(model_data):
        color = COLORS[i % len(COLORS)]
        ax.boxplot(
            scores,
            positions=[i],
            widths=0.55,
            patch_artist=True,
            showfliers=True,
            # medianprops=dict(color="black", linewidth=2.5),
            medianprops=dict(color="#111111", linewidth=2.5),
            boxprops=dict(facecolor=color, color=color, alpha=0.85),
            whiskerprops=dict(color=color, linewidth=1.5),
            capprops=dict(color=color, linewidth=1.5),
            flierprops=dict(
                marker="o",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=4,
                alpha=0.6,
                linewidth=0.5,
            ),
        )

    nice_names = [MODEL_NAMES.get(m, m) for m, _ in model_data]
    ax.set_xticks(positions)
    ax.set_xticklabels(nice_names, rotation=30, ha="right", fontsize=14)
    ax.set_ylabel("Acc@5 (%)", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlim(-0.6, len(model_data) - 0.4)
    ax.set_ylim(0, 105)

    ax.grid(axis="y", color="#d9d9d9", linewidth=1, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_NAME}.png", dpi=300)
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    print("Saved:")
    print(f"{OUTPUT_NAME}.png")
    print(f"{OUTPUT_NAME}.pdf")


if __name__ == "__main__":
    main()