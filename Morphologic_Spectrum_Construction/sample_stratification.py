import pandas as pd
from pathlib import Path
import re
import yaml
from collections import Counter


def classify_stability(prediction_freqs: float) -> str:
    """Classify slide stability based on prediction accuracy frequency."""
    if prediction_freqs >= 0.9:
        return 'Consistently_Correct'
    elif prediction_freqs <= 0.1:
        return 'Consistently_Incorrect'
    else:
        return 'Highly_Variable'


def slide_result_single_model(each_slide_result_csv, slide_results, label_map_short):
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
            slide_results[sid] = {'label': label_map_short[label]}
            slide_results[sid]['pred_label'] = []
            slide_results[sid]['pred_corr'] = []
            slide_results[sid]['misclass'] = []
        # Record prediction result for this slide in this round
        slide_results[sid]['pred_corr'].append(corr)
        slide_results[sid]['pred_label'].append(label_map_short[pred_label])
        # If misclassified, record the misclassified class index
        if corr == 0:
            slide_results[sid]['misclass'].append(pred_label)

    return slide_results


def run_sample_stratification(
    test_result_root,
    metadata_csv,
    out_dir,
    config_path=None,
    glob_pattern="Datasplit_*_10_fold_by_patient*/each_slide_result/*_each_slide_result.csv",
):
    """
    Aggregate per-slide prediction results across all splits/rounds and classify
    slides into stability categories.

    Parameters
    ----------
    test_result_root : str or Path
        Root directory containing all test results (e.g. ./test_result),
        which should include subfolders like:
        Datasplit_*/each_slide_result/*_each_slide_result.csv
    metadata_csv : str or Path
        Cohort metadata CSV file (must include 'Person ID', 'Tissue ID',
        and 'SVS Filename' columns).
    out_dir : str or Path
        Output directory where 'sample_stratification_result.csv' will be written.
    config_path : str or Path or None, optional
        Path to YAML config file that defines LABEL_MAP_SHORT, e.g.
        Morphologic_Spectrum_Construction/config.yaml.
        If None, defaults to `config.yaml` next to this script.
    glob_pattern : str, optional
        Glob pattern (relative to test_result_root) for locating per-slide result
        CSV files. Defaults to
        'Datasplit_*_10_fold_by_patient*/each_slide_result/*_each_slide_result.csv'.

    Returns
    -------
    pandas.DataFrame
        The final merged stratification DataFrame (also written to out_dir).
    """
    test_result_root = Path(test_result_root)
    metadata_csv = Path(metadata_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load config & LABEL_MAP_SHORT ----------
    if config_path is None:
        this_dir = Path(__file__).resolve().parent
        config_path = this_dir / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    label_map_short = config.get("LABEL_MAP_SHORT", {})

    # ---------- Collect per-slide result CSVs ----------
    csv_files = [
        f.resolve()
        for f in test_result_root.glob(glob_pattern)
    ]
    # Sort by split index in the directory name (Datasplit_0, Datasplit_1, ...)
    csv_files = sorted(
        csv_files,
        key=lambda x: int(
            re.search(r"Datasplit_(\d+)_10_fold_by_patient", str(x)).group(1)
        ),
    )

    if not csv_files:
        raise RuntimeError(
            f"[ERROR] No per-slide result CSVs found under {test_result_root} "
            f"matching pattern: {glob_pattern}"
        )

    csv_files_paths = "\n".join([str(file) for file in csv_files])
    print(f"[INFO] Collected each_slide_result.csv files:\n{csv_files_paths}")

    # ---------- Aggregate predictions for each slide ----------
    slide_results = {}
    for csv_path in csv_files:
        slide_results = slide_result_single_model(
            str(csv_path), slide_results, label_map_short
        )

    # ---------- Build DataFrame with per-slide statistics ----------
    rows = []
    for slide_id, results in slide_results.items():
        row = {"slide_id": slide_id, "label": results["label"]}

        # store predicted class names for each round as pred_round_0, pred_round_1, ...
        pred_labels = results["pred_label"]
        for i, pred_lab in enumerate(pred_labels):
            row[f"pred_round_{i}"] = pred_lab

        predictions = results["pred_corr"]
        print(f"[DEBUG] {slide_id} predictions: {predictions}")

        if len(predictions) == 0:
            p_percent = 0.0
        else:
            p_percent = sum(predictions) / len(predictions)

        row["pred_acc"] = p_percent

        # count misclassification frequencies (by numeric class id)
        misclass_ids = results.get("misclass", [])
        misclass_counter = Counter(misclass_ids)
        misclass_str = ", ".join(
            f"{cls}({cnt})" for cls, cnt in misclass_counter.items()
        )
        row["misclass"] = misclass_str

        # assign stability category
        row["stability"] = classify_stability(p_percent)
        rows.append(row)

    current_df = pd.DataFrame(rows)

    # ---------- Merge patient metadata ----------
    person_df = pd.read_csv(metadata_csv)

    # derive slide_id from SVS filename (remove .svs suffix)
    person_df["slide_id"] = person_df["SVS Filename"].str.replace(".svs", "", regex=False)
    person_df = person_df.rename(columns={"Person ID": "person_id"})[
        ["person_id", "Tissue ID", "slide_id"]
    ]

    merged_df = current_df.merge(person_df, on="slide_id", how="left")

    # ---------- Save final stratification results ----------
    slide_track_result_path = out_dir / "sample_stratification_result.csv"
    merged_df.to_csv(slide_track_result_path, index=False, encoding="utf_8_sig")
    print(f"[INFO] Stratification results written to: {slide_track_result_path}")

    return merged_df


def main(args):
    run_sample_stratification(
        test_result_root=args.test_result_root,
        metadata_csv=args.metadata_csv,
        out_dir=args.out_dir,
        config_path=args.config,
    )

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
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (e.g. Morphologic_Spectrum_Construction/config.yaml). "
            "If not provided, defaults to config.yaml next to this script.",
    )
    args = parser.parse_args()

    main(args)
