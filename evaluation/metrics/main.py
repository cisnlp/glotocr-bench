import os
import glob
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from configs import MODELS, DATASET_ROOT, RESULT_DIR, EXTRACTORS
from utils import (
    normalize_text,
    cer,
    detect_script,
    save_json,
)


def load_model_dataset(model_dir):

    print("  • Searching parquet files")

    parquet_files = glob.glob(os.path.join(model_dir, "*.parquet"))

    print(f"  • Found {len(parquet_files)} parquet files")

    dfs = []

    for p in tqdm(parquet_files, desc="  • Loading parquet"):
        dfs.append(pd.read_parquet(p))

    df = pd.concat(dfs, ignore_index=True)

    print(f"  • Loaded {len(df)} rows")

    return df


def evaluate_model(model):

    print("\n==============================")
    print("Evaluating model:", model)
    print("==============================")

    model_dir = os.path.join(DATASET_ROOT, model)

    print("Stage 1: Loading dataset")
    df = load_model_dataset(model_dir)

    extractor = EXTRACTORS[model]

    print("Stage 2: Running evaluation")

    script_metrics = defaultdict(lambda: {
        "samples": 0,
        "cer_sum": 0.0,
        "exact": 0,
        "acc02": 0,
        "acc10": 0,
        "script_correct": 0,
    })

    script_confusion = defaultdict(lambda: defaultdict(int))

    language_metrics = defaultdict(lambda: {
        "samples": 0,
        "cer_sum": 0.0,
        "exact": 0,
        "acc02": 0,
        "acc10": 0
    })

    major_scripts = {"Latn", "Arab", "Cyrl", "Deva"}

    total_samples = 0
    total_exact = 0
    total_acc02 = 0
    total_acc10 = 0
    total_script_correct = 0
    total_cer = 0

    script_examples = defaultdict(list)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  • Evaluating samples"):

        gt_raw = row["text"]
        gt = normalize_text(gt_raw)

        pred_raw = row["markdown"]
        pred = extractor(pred_raw)

        script = row["script"]
        language = row["language"]

        c = cer(pred, gt)

        pred_script = detect_script(pred)

        total_samples += 1

        script_metrics[script]["samples"] += 1
        script_metrics[script]["cer_sum"] += c
        total_cer += c

        if c == 0:
            total_exact += 1
            script_metrics[script]["exact"] += 1

        if c <= 0.02:
            total_acc02 += 1
            script_metrics[script]["acc02"] += 1

        if c <= 0.10:
            total_acc10 += 1
            script_metrics[script]["acc10"] += 1

        if pred_script == script:
            total_script_correct += 1
            script_metrics[script]["script_correct"] += 1

        # Script confusion tracking
        script_confusion[script][pred_script] += 1

        # Language metrics for major scripts
        if script in major_scripts:

            language_metrics[(script, language)]["samples"] += 1
            language_metrics[(script, language)]["cer_sum"] += c

            if c == 0:
                language_metrics[(script, language)]["exact"] += 1

            if c <= 0.02:
                language_metrics[(script, language)]["acc02"] += 1

            if c <= 0.10:
                language_metrics[(script, language)]["acc10"] += 1

        # Save examples for debugging
        script_examples[script].append({
            "gt_raw": gt_raw,
            "pred_raw": pred_raw,
            "gt_norm": gt,
            "pred_norm": pred,
            "cer": c
        })

    print("Stage 3: Computing per-script metrics")

    script_results = {}

    for s, m in script_metrics.items():

        script_results[s] = {
            "samples": m["samples"],
            "CER": m["cer_sum"] / m["samples"],
            "ExactAcc": m["exact"] / m["samples"],
            "Acc@0.02": m["acc02"] / m["samples"],
            "Acc@0.10": m["acc10"] / m["samples"],
            "ScriptAcc": m["script_correct"] / m["samples"],
        }

    print("Stage 4: Computing language metrics")

    language_results = defaultdict(dict)

    for (script, lang), m in language_metrics.items():

        language_results[script][lang] = {
            "samples": m["samples"],
            "CER": m["cer_sum"] / m["samples"],
            "ExactAcc": m["exact"] / m["samples"],
            "Acc@0.02": m["acc02"] / m["samples"],
            "Acc@0.10": m["acc10"] / m["samples"],
        }

    print("Stage 5: Computing macro metrics")

    macro_cer = sum(v["CER"] for v in script_results.values()) / len(script_results)
    macro_exact = sum(v["ExactAcc"] for v in script_results.values()) / len(script_results)
    macro_acc02 = sum(v["Acc@0.02"] for v in script_results.values()) / len(script_results)
    macro_acc10 = sum(v["Acc@0.10"] for v in script_results.values()) / len(script_results)
    macro_script_acc = sum(v["ScriptAcc"] for v in script_results.values()) / len(script_results)

    print("Stage 6: Computing micro metrics")

    micro_cer = total_cer / total_samples
    micro_exact = total_exact / total_samples
    micro_acc02 = total_acc02 / total_samples
    micro_acc10 = total_acc10 / total_samples
    micro_script_acc = total_script_correct / total_samples

    summary = {
        "model": model,
        "samples": total_samples,
        "macro": {
            "CER": macro_cer,
            "ExactAcc": macro_exact,
            "Acc@0.02": macro_acc02,
            "Acc@0.10": macro_acc10,
            "ScriptAcc": macro_script_acc
        },
        "micro": {
            "CER": micro_cer,
            "ExactAcc": micro_exact,
            "Acc@0.02": micro_acc02,
            "Acc@0.10": micro_acc10,
            "ScriptAcc": micro_script_acc
        }
    }

    print("Stage 7: Saving results")

    model_result_dir = os.path.join(RESULT_DIR, model)

    os.makedirs(model_result_dir, exist_ok=True)

    save_json(os.path.join(model_result_dir, "summary.json"), summary)
    save_json(os.path.join(model_result_dir, "script_metrics.json"), script_results)
    save_json(os.path.join(model_result_dir, "script_confusion.json"), script_confusion)

    for script in major_scripts:

        if script in language_results:

            save_json(
                os.path.join(model_result_dir, f"language_metrics_{script}.json"),
                language_results[script]
            )

    print("Stage 8: Saving script debug CSVs")

    scripts_dir = os.path.join(model_result_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    for script, rows in script_examples.items():

        df_script = pd.DataFrame(rows)

        out_path = os.path.join(scripts_dir, f"{script}.csv")

        df_script.to_csv(out_path, index=False)

    print("✔ Finished model:", model)


def main():

    print("Starting UniOCR evaluation")

    os.makedirs(RESULT_DIR, exist_ok=True)

    for model in MODELS:
        evaluate_model(model)


if __name__ == "__main__":
    main()