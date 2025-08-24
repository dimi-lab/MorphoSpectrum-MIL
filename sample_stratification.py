import pandas as pd
from pathlib import Path
import re
import yaml
from collections import Counter

# ---------------------
# Global constants
# ---------------------
with open("config.yaml","r") as f:
    config = yaml.safe_load(f)
LABEL_MAP_SHORT = config["LABEL_MAP_SHORT"]


def classify_stability(prediction_freqs: float) -> str:
    """Classify slide stability based on prediction accuracy frequency."""
    if prediction_freqs >= 0.9:
        return 'Consistently_Correct'
    elif prediction_freqs <= 0.1:
        return 'Consistently_Incorrect'
    else:
        return 'Highly_Variable'


def slide_result_single_model(each_slide_result_csv, slide_results):
    """
    Aggregate prediction correctness and misclassification trends for each slide.

    Example slide_results structure:
    {
        "slide_001": {
            "label": "POP-NR",
            "pred_corr": [1, 0, 1, 1, 1, 0],   # correctness (1=correct, 0=incorrect) across rounds
            "pred_label": ["POP-NR", "CAP", ...], # predicted class names across rounds
            "misclass": ["CAP", "POP-R"],      # misclassified target classes
        },
        ...
    }
    """
    df = pd.read_csv(each_slide_result_csv)

    # Accumulate predictions for each slide
    for _, row in df.iterrows():
        sid = row['slide_id']
        label = int(row['label'])
        corr = 1 if row['correctness'] is True else 0
        pred_label = int(row['pred'])  # predicted class index
        if sid not in slide_results:
            slide_results[sid] = {'label': LABEL_MAP_SHORT[label]}
            slide_results[sid]['pred_label'] = []
            slide_results[sid]['pred_corr'] = []
            slide_results[sid]['misclass'] = []
        # Record prediction result for this slide in this round
        slide_results[sid]['pred_corr'].append(corr)
        slide_results[sid]['pred_label'].append(LABEL_MAP_SHORT[pred_label])
        # If misclassified, record the misclassified class index
        if corr == 0:
            slide_results[sid]['misclass'].append(pred_label)

    return slide_results


def main(args):
    # Initialize output file path
    slide_track_result_path = Path(args.out_dir) / "sample_stratification_result.csv"
    slide_results = {}

    # Collect per-slide prediction result CSVs from different splits
    test_result_basepath = Path(args.test_result_root)
    csv_files = [
        f.resolve()
        for f in test_result_basepath.glob(
            "Datasplit_*_10_fold_by_patient_random/each_slide_result/*_each_slide_result.csv"
        )
    ]

    # Sort CSV files by split index (Datasplit_0, Datasplit_1, ...)
    csv_files = sorted(
        csv_files,
        key=lambda x: int(
            re.search(r"Datasplit_(\d+)_10_fold_by_patient_random", str(x)).group(1)
        ),
    )

    csv_files_paths = "\n".join([str(file) for file in csv_files])
    print(f"[INFO] Collected each_slide_result.csv files:\n{csv_files_paths}")

    # Process each result CSV
    for csv in csv_files:
        slide_results = slide_result_single_model(str(csv), slide_results)

    # Build a DataFrame with per-slide prediction statistics
    rows = []
    for slide_id, results in slide_results.items():
        row = {"slide_id": slide_id, "label": results["label"]}

        # Store predicted class names for each round
        pred_labels = results["pred_label"]
        for i, pred_lab in enumerate(pred_labels):
            row[f"pred_round_{i}"] = pred_lab

        predictions = results["pred_corr"]
        print(f"[DEBUG] {slide_id} predictions: {predictions}")

        # Compute prediction accuracy across rounds
        p_percent = sum(predictions) / len(predictions)
        row["pred_acc"] = p_percent

        # Count misclassification frequencies
        misclass = results.get("misclass", [])
        misclass_counter = Counter(misclass)
        misclass_str = ", ".join([f"{cls}({cnt})" for cls, cnt in misclass_counter.items()])
        row["misclass"] = misclass_str

        # Assign stability category
        row["stability"] = classify_stability(p_percent)
        rows.append(row)

    current_df = pd.DataFrame(rows)

    # ----------------- Merge patient metadata ----------------- #
    # Read metadata CSV (slide-level mapping to patient and tissue IDs)
    person_info_path = Path(args.metadata_csv)
    person_df = pd.read_csv(person_info_path)

    # Clean SVS filename (remove .svs suffix) and rename columns
    person_df["slide_id"] = person_df["SVS Filename"].str.replace(".svs", "", regex=False)
    person_df = person_df.rename(columns={"Person ID": "person_id"})[
        ["person_id", "Tissue ID", "slide_id"]
    ]

    # Merge with current per-slide statistics
    merged_df = current_df.merge(person_df, on="slide_id", how="left")

    # Save final stratification results
    merged_df.to_csv(slide_track_result_path, index=False, encoding="utf_8_sig")
    print(f"[INFO] Stratification results written to: {slide_track_result_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample stratification script: aggregate per-slide predictions across CV rounds and classify stability.")
    parser.add_argument(
        "--test_result_root",
        type=str,
        default="./test_result",
        help="Root directory containing all test results generated during MIL model evaluation (e.g., ./test_result).",
        required=True,
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        help="Path to the cohort metadata CSV (should include 'Person ID', 'Tissue ID', and 'SVS Filename' columns).",
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory where the aggregated stratification CSV will be stored.",
        required=True,
    )
    args = parser.parse_args()

    main(args)
