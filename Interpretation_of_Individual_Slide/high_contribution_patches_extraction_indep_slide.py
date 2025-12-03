import numpy as np
import logging
import seaborn as sns
import sys
import pandas as pd
from pathlib import Path
import csv
import argparse
import json
import shutil
import os
import openslide
import matplotlib.pyplot as plt
import h5py
import yaml
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------- Model import (project layout aware) ----------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from Morphologic_Spectrum_Construction.model import AttentionNet

# ---------------------
# Global constants
# ---------------------

SCALE_MARKER = {
    "tissue": "o",       # circle
    "cluster": "x",      # cross
    "cell": "s",         # square
}
DEVICE = torch.device("cuda")



# --------------------- Model loading --------------------- #
def load_trained_model(device, checkpoint_path, model_size, input_feature_size, n_classes):
    model = AttentionNet(
        input_feature_size=input_feature_size,
        n_classes=n_classes,
        model_size=model_size,
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# --------------------- Attention evaluation per class --------------------- #
def evaluate_att_scores_single_slide(target_class_idx, feature_bag_path,model,device):
    """
    Compute attention scores on a slide for a given target_class_idx.
    Returns attention scores, weights, feature embeddings, and tile IDs.
    """

    att_score_slide = {}

    test_features, tile_ids = getCONCHFeatures(feature_bag_path)
    att_score_slide["features"] = test_features.numpy()

    test_features = test_features.to(device)
    A, h, classifiers, Y_prob, Y_hat = predict_attention_matrix(model, test_features)

    # Attention score for this target class
    Atten_scale = A[target_class_idx]
    att_score = Atten_scale * np.dot(h, classifiers[target_class_idx])
    att_score_slide["att_score"] = att_score
    att_score_slide["att_weight"] = A[target_class_idx]
    att_score_slide["subtype"] = int(target_class_idx)

    return att_score_slide, tile_ids, Y_prob, int(Y_hat)


# --------------------- H5 utilities --------------------- #
def getCONCHFeatures(feature_bag_path):
    try:
        features, _, tile_ids = split_feature_tile_id(feature_bag_path)
        print(f"[DEBUG] Feature shape: {features.shape}")
        print(f"[DEBUG] Number of tile IDs: {len(tile_ids)}")
    except KeyError:
        raise KeyError(f"The file {feature_bag_path} does not contain valid datasets.")

    features = torch.from_numpy(features)
    return features, tile_ids


def split_feature_tile_id(h5_path):
    with h5py.File(h5_path, "r") as file:
        features_all = file["features"][:]
        tile_ids_all = file["tile_ids"][:]
        coords_all   = file["coords"][:]

    # Convert tile_ids to Python strings
    tile_ids_str = [
        tid.decode("utf-8") if isinstance(tid, (bytes, bytearray)) else str(tid)
        for tid in tile_ids_all
    ]

    feats, coords, tile_ids = [], [], []

    for feature, coord, tid in zip(features_all, coords_all, tile_ids_str):
        feats.append(feature)
        coords.append(coord)
        tile_ids.append(tid)

    feats = np.array(feats, dtype=np.float32)
    return feats, coords, tile_ids


# --------------------- Model forward pass --------------------- #
def predict_attention_matrix(model, feature_batch):
    with torch.no_grad():
        logits, Y_prob, Y_hat, A_raw, M, h, classifiers = model(feature_batch)
        A = F.softmax(A_raw, dim=1)
    return (
        A.cpu().numpy(),
        h.cpu().numpy(),
        classifiers.cpu().numpy(),
        Y_prob.cpu().numpy(),
        Y_hat.item(),
    )


# --------------------- Top-K selection --------------------- #
def topk_by_cumsum(scores, ratio=0.9):
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    cumulative = np.cumsum(sorted_scores)

    if cumulative[-1] <= 0:
        return sorted_indices[:1]

    threshold = ratio * cumulative[-1]
    k = np.searchsorted(cumulative, threshold) + 1
    return sorted_indices[:k]


# --------------------- Extract patches for all predictions --------------------- #
def process_slide_for_all_preds(h5_path,model, device, n_classes, topk_ratio):
    results = []

    for pred_idx in range(n_classes):
        att_score_slide, tile_ids, _, _ = evaluate_att_scores_single_slide(
            target_class_idx=pred_idx,
            feature_bag_path=h5_path,
            model=model,
            device=device,
        )

        scores = att_score_slide["att_score"]
        top_k_centers = topk_by_cumsum(scores, ratio=topk_ratio)

        top_k_ids = list(np.array(tile_ids)[top_k_centers])
        all_scores  = att_score_slide["att_score"]
        all_weights = att_score_slide["att_weight"]
        all_feats   = att_score_slide["features"]

        top_k_scores   = all_scores[top_k_centers]
        top_k_weights  = all_weights[top_k_centers]
        top_k_features = all_feats[top_k_centers]

        results.append((pred_idx, top_k_ids, top_k_scores, top_k_weights, top_k_features))

    return results


def run_high_contri_extraction(
    config_path: str,
    manifest_path: Path,
    attn_checkpoint: Path,
    h5_base_path: Path,
    output_dir: Path,
    topk_ratio: float = 0.9,
    slide_ids: Optional[Iterable[str]] = None,
) -> Path:
    """
    Extract high-contribution patches for all independent slides.

    Args:
        config: loaded YAML config dict (used for label mappings, N_CLASSES, etc.).
        manifest_path: CSV listing independent slides (contains 'SVS Filename', 'Subtype', ...).
        attn_checkpoint: path to full-data AttentionNet checkpoint.
        h5_base_path: directory containing {slide_id}_features.h5 files.
        output_dir: directory where high-contribution patch CSVs will be written.
        device: 'cuda' or 'cpu'.
        topk_ratio: cumulative contribution ratio for selecting top-K patches (default 0.9).
        slide_ids: optional iterable of slide_ids to restrict processing; if None, process all slides.

    Returns:
        Path to unified CSV file: output_dir / 'high_attened_patches_indep_data.csv'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    # ------ Static settings from config ------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    n_classes = int(config["N_CLASSES"])
    label_map_short: Dict[int, str] = config["LABEL_MAP_SHORT"]
    label_long_short: Dict[str, str] = config["subtypes"]["long_to_short"]
    input_feature_size = int(config["algorithm"]["input_feature_size"])

    # ------ Load model once ------
    print(f"[INFO] Loading AttentionNet from: {attn_checkpoint}")
    model = load_trained_model(
        device=DEVICE,
        checkpoint_path=Path(attn_checkpoint),
        model_size="small",
        input_feature_size=input_feature_size,
        n_classes=n_classes,
    )

    # ------ Open unified CSV writer ------
    coords_csv_path = output_dir / "high_attened_patches_indep_data.csv"
    with open(coords_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["slide_id", "label", "pred", "tile_id", "weight", "score", "embed"])

        # Load manifest CSV
        result_df = pd.read_csv(manifest_path)

        # Optional: restrict to selected slide_ids
        if slide_ids is not None:
            slide_ids_set = set(str(s) for s in slide_ids)
            result_df = result_df[result_df["SVS Filename"].apply(
                lambda x: str(x).replace(".svs", "") in slide_ids_set
            )]

        for _, row in result_df.iterrows():
            slide_id = str(row["SVS Filename"]).replace(".svs", "")
            subtype_long = str(row["Subtype"]).strip()

            try:
                label_short = label_long_short[subtype_long]
            except Exception:
                raise ValueError(
                    f"Subtype '{subtype_long}' not found in config['subtypes']['long_to_short']."
                )

            h5_path = Path(h5_base_path) / f"{slide_id}_features.h5"

            print("\n" + "=" * 50)
            print(
                f"[START] Processing slide {slide_id} from {h5_path} — "
                f"extracting high-contribution patches for each predicted class"
            )

            per_pred_results = process_slide_for_all_preds(
                h5_path=h5_path,
                model=model,
                device=DEVICE,
                n_classes=n_classes,
                topk_ratio=topk_ratio,
            )

            for (pred_idx, top_k_ids, top_k_scores, top_k_weights, top_k_features) in per_pred_results:
                print(f"pred_idx={pred_idx}")
                pred_label_short = label_map_short[pred_idx] if isinstance(label_map_short, dict) else label_map_short[pred_idx]
                for i, item_id in enumerate(top_k_ids):
                    feature_vec = top_k_features[i]
                    feature_str = ";".join(map(str, feature_vec))
                    weight = float(top_k_weights[i])
                    atten_score = float(top_k_scores[i])
                    writer.writerow(
                        [
                            slide_id,
                            label_short,
                            pred_label_short,
                            item_id,
                            weight,
                            atten_score,
                            feature_str,
                        ]
                    )

            print(f"[DONE] Slide {slide_id} completed")

    # ------ Split unified CSV by label ------
    df_all = pd.read_csv(coords_csv_path)
    for label, subset in df_all.groupby("label"):
        out_name = output_dir / f"independent_{label}.csv"
        subset.to_csv(out_name, index=False)
        print(f"[INFO] Saved {out_name}, rows={len(subset)}")

    return coords_csv_path

# --------------------- Main --------------------- #
def main(args):

    run_high_contri_extraction(
        config=args.config,
        manifest_path=Path(args.manifest),
        attn_checkpoint=Path(args.attn_checkpoint),
        h5_base_path=Path(args.h5_base),
        output_dir=Path(args.output_dir),
        topk_ratio=args.topk_ratio,
        slide_ids=args.slides,
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract high-contribution patches for independent slides."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config.yaml (for label mappings, algorithm params).")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to independent_svs_file_mapping.csv.")
    parser.add_argument("--attn-checkpoint", type=str, required=True,
                        help="Path to full-data AttentionNet checkpoint (.pt).")
    parser.add_argument("--h5-base", type=str, required=True,
                        help="Directory containing {slide_id}_features.h5 files.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save high-contribution patch CSVs.")
    parser.add_argument("--topk-ratio", type=float, default=0.9,
                        help="Cumulative contribution ratio for selecting top-K patches.")
    parser.add_argument(
        "--slides",
        nargs="*",
        default=None,
        help="Optional list of slide_ids to process (default: all slides in manifest).",
    )
    args = parser.parse_args()
    main(args)
