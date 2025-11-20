# Generate cross-validation splits at the patient level.
# - Groups slides by patient to avoid data leakage.
# - Two assignment strategies are supported:
#   1) "even": greedily balances the number of slides per fold by patient slide counts.
#   2) "random": randomly assigns patients to folds (round-robin after shuffle).
# - For each subtype, produces n_folds folds; for each round,
#   selects train/val/test folds based on provided ratios and marks each slide.
#
# Outputs a single CSV per shuffle with columns:
#   Person ID, slide_id, class (Subtype), label, round-0 ... round-(n_folds-1)
#
# Label mapping is loaded from a YAML config (see --label_config).

import argparse
import random
from pathlib import Path

import pandas as pd
import yaml


def load_label_config(path: str):
    """Load label config from YAML and build name<->id mappings."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    id2name = cfg.get("LABEL_MAP", {})
    name2id = {v: int(k) for k, v in id2name.items()}
    short_names = cfg.get("LABEL_MAP_SHORT", {})
    if len(id2name) != cfg.get("N_CLASSES", len(id2name)):
        print("[WARN] N_CLASSES does not match LABEL_MAP length:",
              cfg.get("N_CLASSES"), "vs", len(id2name))
    return {
        "n_classes": cfg.get("N_CLASSES", len(id2name)),
        "id2name": id2name,
        "name2id": name2id,
        "short": short_names,
    }


def split_persons_randomly(subtype: str, person_list, n_folds: int, seed: int):
    """
    Shuffle patients and assign to folds in a round-robin manner.
    This does not strictly enforce slide-count balance per fold.
    """
    random.seed(seed)
    random.shuffle(person_list)
    print(f"{subtype} shuffled patients (pid, #slides): "
          f"{[(pid, len(pdf)) for pid, pdf in person_list]}")

    folds_person = [[] for _ in range(n_folds)]
    fold_slide_counts = [0] * n_folds

    for idx, (pid, pdf) in enumerate(person_list):
        fold_idx = idx % n_folds
        folds_person[fold_idx].append((pid, pdf))
        fold_slide_counts[fold_idx] += len(pdf)

    for fold_idx in range(n_folds):
        print(f"{subtype} Fold {fold_idx}: total slides {fold_slide_counts[fold_idx]}, "
              f"patients: {[(pid, len(pdf)) for pid, pdf in folds_person[fold_idx]]}")
    return folds_person


def split_persons_by_slide_count(subtype: str, person_list, n_folds: int, seed: int):
    """
    Assign patients to folds by sorting patients (ascending by #slides) and
    greedily filling each fold to keep slide counts roughly balanced.
    A subtype-specific threshold is used to allow slight overfill.
    """
    random.seed(seed)

    # Total number of slides for this subtype
    total_slides = sum(len(pdf) for (_, pdf) in person_list)
    print(f"{subtype} total slides: {total_slides}")
    size_per_fold = total_slides // n_folds
    print(f"{subtype} target slides per fold (floor): {size_per_fold}")

    # Subtype-specific threshold that allows slight overfill per fold
    if subtype == "Clear_Cell_Carcinoma":
        threshold = 0
    elif subtype == "Ovarian_Endometrioid_Carcinoma":
        threshold = 1
    else:
        threshold = 2

    # Sort patients from fewest to most slides
    person_list.sort(key=lambda x: len(x[1]))
    print(f"{subtype} patients sorted by #slides (pid, #slides): "
          f"{[(pid, len(pdf)) for pid, pdf in person_list]}")

    folds_person = [[] for _ in range(n_folds)]
    current_fold_idx = 0
    current_fold_slides = 0

    for (pid, pdf) in person_list:
        p_slides = len(pdf)
        # Fill current fold if capacity allows; otherwise move to next fold
        if (current_fold_slides + p_slides <= size_per_fold + threshold) or (current_fold_idx == n_folds - 1):
            folds_person[current_fold_idx].append((pid, pdf))
            current_fold_slides += p_slides
        else:
            current_fold_idx += 1
            current_fold_slides = 0
            folds_person[current_fold_idx].append((pid, pdf))
            current_fold_slides += p_slides

    for fold_idx in range(n_folds):
        print(f"{subtype} Fold {fold_idx} patients (pid, #slides): "
              f"{[(pid, len(pdf)) for pid, pdf in folds_person[fold_idx]]}")
    return folds_person


def multi_scale_split(
    subtype: str,
    df_subtype: pd.DataFrame,
    n_folds: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    mode: str,
    name2id: dict,
):
    """
    Build patient-level folds for a subtype and mark each slide's role (train/val/test)
    for every round in [0, n_folds-1]. Each round rotates which folds serve as train/val/test.
    """
    # Compute counts of folds for train/val/test
    train_count = int(round(n_folds * train_ratio))
    val_count = int(round(n_folds * val_ratio))
    test_count = int(round(n_folds * test_ratio))
    print(f"Fold allocation: train {train_count}, val {val_count}, test {test_count} (total {n_folds})\n")

    # Define rotating fold sets for each round
    n_rounds = n_folds
    train_folds, val_folds, test_folds = [], [], []
    for round_idx in range(n_rounds):
        train_folds.append([(round_idx + j) % n_folds for j in range(train_count)])
        val_folds.append([(round_idx + train_count + j) % n_folds for j in range(val_count)])
        test_folds.append([(round_idx + train_count + val_count + j) % n_folds for j in range(test_count)])

    print(f"train_folds: {train_folds}")
    print(f"val_folds:   {val_folds}")
    print(f"test_folds:  {test_folds}")

    # Group slides by patient
    person_groups = df_subtype.groupby("Person ID")
    person_list = [(pid, pdf) for pid, pdf in person_groups]

    # Build folds at the patient level
    if mode == "even":
        folds_person = split_persons_by_slide_count(subtype, person_list, n_folds, seed)
    elif mode == "random":
        folds_person = split_persons_randomly(subtype, person_list, n_folds, seed)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'even' or 'random'.")

    # Initialize output records for each slide in this subtype
    results = {
        row["slide_id"]: {
            "Person ID": row["Person ID"],
            "slide_id": row["slide_id"],
            "class": row["Subtype"],
            "label": name2id.get(row["Subtype"], -1),
        }
        for _, row in df_subtype.iterrows()
    }

    if any(v["label"] == -1 for v in results.values()):
        print("[WARN] Found subtype not present in LABEL_MAP; label=-1 assigned. Check label_config and data.")

    for result in results.values():
        for round_idx in range(n_rounds):
            result[f"round-{round_idx}"] = "unknown"

    # Mark each slide's role per round according to its fold assignment
    for round_idx in range(n_rounds):
        for fold_idx in range(n_folds):
            for (pid, pdf) in folds_person[fold_idx]:
                for _, row in pdf.iterrows():
                    slide_id = row["slide_id"]
                    if fold_idx in train_folds[round_idx]:
                        results[slide_id][f"round-{round_idx}"] = "training"
                    elif fold_idx in val_folds[round_idx]:
                        results[slide_id][f"round-{round_idx}"] = "validation"
                    elif fold_idx in test_folds[round_idx]:
                        results[slide_id][f"round-{round_idx}"] = "testing"
                    else:
                        results[slide_id][f"round-{round_idx}"] = "undefined"

    return results


def main(args):
    # Load dataset CSV
    csv_file_path = args.data_csv
    df = pd.read_csv(csv_file_path)

    # Ensure output directory exists
    output_csv_dir = Path(args.output_dir)
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    # Read core parameters
    n_splits = args.split_num
    n_folds = args.fold_num
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio

    # Simple ratio check
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("Data ratio setting error: train_ratio + val_ratio + test_ratio must equal 1")

    # Create per-shuffle seeds
    seed_list = random.sample(range(1000), n_splits)

    # Derive slide_id from SVS filename (remove brackets/quotes and .svs suffix)
    df["slide_id"] = df["SVS Filename"].str.replace(".svs", "")

    # Load label mapping config
    label_cfg = load_label_config(args.label_config)
    name2id = label_cfg["name2id"]

    # Mode: "even" or "random"
    mode = args.mode

    # Split by subtype and process each subtype independently
    grouped = df.groupby("Subtype")

    for s_index in range(n_splits):
        print(f"\nBegin split #{s_index}")
        all_results = {}
        for subtype, df_subtype in grouped:
            print(f"\nStart {subtype} splitting into {n_folds} folds")
            subtype_results = multi_scale_split(
                subtype=subtype,
                df_subtype=df_subtype,
                n_folds=n_folds,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed_list[s_index],
                mode=mode,
                name2id=name2id,
            )
            all_results |= subtype_results

        # Write one CSV per shuffle
        csv_filename = f"Datasplit_{s_index}_{n_folds}_fold_by_patient.csv"
        output_csv_path = output_csv_dir / csv_filename
        final_df = pd.DataFrame(list(all_results.values()))
        final_df.to_csv(output_csv_path, index=False)
        print(f"Wrote split CSV: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient-level data splitting for cross-validation")
    parser.add_argument(
        "--mode",
        choices=["even", "random"],
        help="'even': balance slides per fold by patient slide counts; 'random': assign patients randomly to folds",
        required=True,
    )
    parser.add_argument(
        "--split_num",
        type=int,
        help="Number of independent shuffles",
        required=True,
    )
    parser.add_argument(
        "--fold_num",
        type=int,
        help="Number of folds for cross-validation",
        required=True,
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        help="Training fold ratio; requires train_ratio + val_ratio + test_ratio = 1",
        required=True,
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        help="Validation fold ratio",
        required=True,
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        help="Test fold ratio",
        required=True,
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        help="Path to the dataset CSV file",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to store the generated split CSVs",
        required=True,
    )
    parser.add_argument(
        "--label_config",
        type=str,
        help="Path to YAML for subtype/label mapping (e.g., configs/labels.yaml)",
        required=True,
    )
    args = parser.parse_args()
    main(args)
