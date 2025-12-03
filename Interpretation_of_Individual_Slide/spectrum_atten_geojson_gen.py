#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3 – Attention Heatmap + Spectrum-based Local Explanation (Single Slide)

Single-slide pipeline:
1) Read patch assignment CSV (output of spectrum mapping), join H5 coords → write tiles CSV.
2) Run the attention model once on the slide, combine:
      - per-class attention tiles
      - spectrum-cluster annotations
   → write ONE GeoJSON file for MorphoExplainer.

This module is designed as a library:
- It does NOT parse command-line arguments.
- Use the top-level function `run_single_slide_explanation(...)`
  from a notebook or another Python script.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image
from shapely.affinity import scale as shp_scale
from shapely.geometry import box

# ---------------- Model import (project layout aware) ----------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from Morphologic_Spectrum_Construction.model import AttentionNet 


# ---------------- Pretty logging ----------------
def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# ---------------- Small helpers ----------------
def infer_slide_id(slide_path: Path) -> str:
    return slide_path.stem


def rglob_first(base: Path, pattern: str) -> Optional[Path]:
    matches = list(base.rglob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        logging.warning(f"[rglob] Multiple matches for pattern '{pattern}', using first: {matches[0]}")
    return matches[0]


def standardize_scores(raw_1d: np.ndarray) -> np.ndarray:
    z = (raw_1d - np.mean(raw_1d)) / (np.std(raw_1d) + 1e-12)
    z = z + np.abs(np.min(z))
    maxv = np.max(z)
    if maxv > 0:
        z = z / maxv
    return z.astype(np.float32)


def rect_to_polygon(x, y, w, h):
    x, y, w, h = map(float, (x, y, w, h))
    return [[[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]]


def scale_rectangles(raw_rect_bounds: np.ndarray, scale_factor: float):
    rects = []
    for coords in raw_rect_bounds:
        minx, miny, maxx, maxy = coords
        rect = box(minx, miny, maxx, maxy)
        rects.append(shp_scale(rect, xfact=scale_factor, yfact=scale_factor, zfact=1.0, origin=(0, 0)))
    return rects


def get_display_image(wsi, display_level: int) -> Tuple[Image.Image, float]:
    max_level = len(wsi.level_dimensions) - 1
    if display_level >= max_level:
        display_level = max(max_level - 1, 0)
    display_image = wsi.read_region((0, 0), display_level, wsi.level_dimensions[display_level]).convert("RGBA")
    scale_factor = 1 / float(wsi.level_downsamples[display_level])
    return display_image, scale_factor


def extract_hc_name(cluster: str) -> str:
    if not cluster:
        return "hc_unknown"
    s = str(cluster).strip()
    s = s.replace(":", "_").replace("-", "_")
    return s


def hc_index(hc_name: str) -> int:
    m = re.search(r'hc[_-](\d+)$', hc_name)
    return int(m.group(1)) if m else -1


def hc_to_color_index(hc_name: str, palette_len: int) -> int:
    m = re.search(r'hc[_-](\d+)$', hc_name)
    if m:
        return (int(m.group(1)) - 1) % palette_len
    return (abs(hash(hc_name)) % 997) % palette_len


# ---------------- H5 helpers ----------------
def split_feature_tile_id(h5_path: Path):
    with h5py.File(h5_path, "r") as file:
        features_all = file["features"][:]
        tile_ids_all_bytes = file["tile_ids"][:]
        coors_all = file["coords"][:]

    tile_ids_str = [
        tid.decode("utf-8") if isinstance(tid, (bytes, bytearray)) else str(tid) for tid in tile_ids_all_bytes
    ]

    feats, coords, tile_ids = [], [], []
    for feature, coord, tid in zip(features_all, coors_all, tile_ids_str):
        feats.append(feature)
        coords.append(coord)
        tile_ids.append(tid)

    feats = np.array(feats, dtype=np.float32)
    return feats, coords, tile_ids


def get_features_from_h5(h5_path: Path, device: torch.device):
    feats_np, coords_np, tile_ids = split_feature_tile_id(h5_path)
    if feats_np.size == 0:
        raise ValueError(f"[H5] empty bag in {h5_path}")
    features = torch.from_numpy(feats_np).to(device)
    return features, coords_np, tile_ids


# ---------------- Matching helpers ----------------
def build_id2bbox(coords: np.ndarray, ids: List[str]) -> Dict[str, Tuple[int, int, int, int]]:
    id2bbox = {}
    for c, tid in zip(coords, ids):
        minx, miny, maxx, maxy = [int(v) for v in c]
        key = tid.decode("utf-8") if isinstance(tid, (bytes, bytearray)) else str(tid)
        key = key.strip()
        id2bbox[key] = (minx, miny, maxx - minx, maxy - miny)
    return id2bbox


# ---------------- Attention model ----------------
def load_trained_model(
    device: torch.device,
    checkpoint_path: Path,
    model_size: str,
    input_feature_size: int,
    n_classes: int,
):
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


def predict_all_attentions(model: AttentionNet, feature_batch: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits, Y_prob, Y_hat, A_raw, *_ = model(feature_batch)
    A_raw_np = A_raw.detach().cpu().numpy()              # [C, N]
    if Y_prob is not None:
        prob_np = Y_prob.detach().cpu().view(-1).numpy()  # [C]
    else:
        lg = logits.detach().cpu().view(-1).numpy()
        ex = np.exp(lg - np.max(lg))
        prob_np = ex / np.sum(ex)
    return A_raw_np, prob_np


# ---------------- Spectrum labels (human-readable) ----------------
CLASSIFICATION_SETS = {
    "Border:hc_1": "Epithelial Proliferation and Tufting with Papillary Pattern",
    "Border:hc_2": "Simple Layer of Epithelium",
    "Border:hc_3": "Non-neoplastic Tissue",
    "Border:hc_4": "Non-neoplastic Tissue",

    "Clear:hc_1": "Non-neoplastic Tissue",
    "Clear:hc_2": "High Nuclear Atypia; Clear Cytoplasm; Tubulocystic; Hobnail",
    "Clear:hc_3": "High Nuclear Atypia; Clear Cytoplasm; Tubulocystic; Hobnail",
    "Clear:hc_4": "Necrosis",
    "Clear:hc_5": "High Nuclear Atypia; Clear Cytoplasm; Tubulocystic; Hobnail",

    "Endo:hc_1": "Squamous Metaplasia",
    "Endo:hc_2": "Glandular Pattern with Squamous Metaplasia",
    "Endo:hc_3": "Non-neoplastic Tissue",
    "Endo:hc_4": "Glandular Pattern",

    "Endo_to_High:hc_1": "Glandular Pattern with Mitosis/Apoptotic Bodies",
    "Endo_to_High:hc_2": "Solid Pattern",

    "High:hc_1": "Non-neoplastic Tissue",
    "High:hc_2": "Pleomorphism; High N/C Ratio; Nuclear Hyperchromatia",
    "High:hc_3": "Pleomorphism; High N/C Ratio; Nuclear Hyperchromatia",

    "High_to_Endo:hc_1": "Stroma Rich Area With Sparse Tumor Cells",
    "High_to_Endo:hc_2": "Solid Pattern",
    "High_to_Endo:hc_3": "Glandular Pattern",
}

_CLUSTER_RE = re.compile(r'^\s*([A-Za-z_]+)\s*:\s*(hc[_-]\d+)\s*$')


def parse_cluster_full_key(cluster_str: str):
    s = str(cluster_str).strip()
    m = _CLUSTER_RE.match(s)
    if m:
        group = m.group(1).strip()
        hc_name = m.group(2).replace('-', '_').strip()
        return group, hc_name, f"{group}:{hc_name}"
    hc = re.sub(r'-', '_', s)
    return "", hc, hc


def build_classificationsets_from_df(spectrum_df: pd.DataFrame, class_desc_dict: dict) -> list:
    groups, hc_names, full_keys = [], [], []
    clusters = spectrum_df["cluster"].dropna().astype(str)
    for c in clusters:
        g, hc, full = parse_cluster_full_key(c)
        if not hc:
            continue
        groups.append(g)
        hc_names.append(hc)
        full_keys.append(full)
    uniq_full = sorted(
        set(full_keys),
        key=lambda key: int(re.search(r'\d+', key).group()) if re.search(r'\d+', key) else 10**9,
    )
    out = []
    for fk in uniq_full:
        name_fmt = fk.replace(":", "_")
        descr = class_desc_dict.get(fk, "unknown")
        out.append({"name": name_fmt, "descr": descr})
    if not uniq_full and hc_names:
        uniq_hc = sorted(
            set(hc_names),
            key=lambda h: int(re.search(r'\d+', h).group()) if re.search(r'\d+', h) else 10**9,
        )
        for hc in uniq_hc:
            descr = class_desc_dict.get(hc, "unknown")
            out.append({"name": hc, "descr": descr})
    return out


# ---------------- GeoJSON builders ----------------
def make_attention_tile_feature(bounds_tuple, score, slide_id, pred_class_name):
    minx, miny, maxx, maxy = bounds_tuple
    poly = [
        [float(minx), float(miny)],
        [float(maxx), float(miny)],
        [float(maxx), float(maxy)],
        [float(minx), float(maxy)],
        [float(minx), float(miny)],
    ]
    val = float(score) if np.isfinite(score) else 0.0
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [poly]},
        "properties": {
            "isLocked": True,
            "prediction": pred_class_name,
            "objectType": "tile",
            "measurements": {"attention": val},
            "metadata": {"slide_id": slide_id, "coord_unit": "pixel", "wsi_level": 0},
        },
    }


def make_spectrum_annotation_feature(row, palette, pred_class_name, expect_slide_id=None):
    tile_id = str(row["tile_id"]).strip()
    slide_id = str(row["slide_id"]).strip()
    if slide_id.lower().endswith(".svs"):
        slide_id = slide_id[:-4]
    if expect_slide_id is not None and slide_id != expect_slide_id:
        return None

    cluster = str(row["cluster"]).strip()
    x, y, w, h = float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])

    hc_name = extract_hc_name(cluster)
    color = palette[hc_to_color_index(hc_name, len(palette))]
    ms_idx = hc_index(hc_name)

    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": rect_to_polygon(x, y, w, h)},
        "properties": {
            "prediction": pred_class_name,
            "objectType": "annotation",
            "name": tile_id,
            "color": color,
            "classification": {"name": hc_name, "color": color},
            "measurements": [
                {"name": "TileWidth", "value": w},
                {"name": "TileHeight", "value": h},
                {"name": "MorphoSpectrum", "value": float(ms_idx) if ms_idx >= 0 else -1.0},
            ],
            "metadata": {"slide_id": slide_id, "tile_id": tile_id, "coord_unit": "pixel", "wsi_level": 0},
        },
    }


def export_multi_pred_geojson(pred_entries: List[dict], all_features: List[dict], out_path: Path):
    fc = {"type": "FeatureCollection", "predictions": pred_entries, "features": all_features}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    logging.info(f"[GeoJSON] Wrote: {out_path} (preds={len(pred_entries)}, features={len(all_features)})")


# ---------------- Tiles CSV creation (assign → coords) ----------------
def read_assign_csv(assign_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(assign_csv)
    required = {"slide_id", "tile_id", "assigned_cluster", "pred"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"[assign] Missing columns {sorted(miss)} in {assign_csv}")
    df = df.rename(columns={"assigned_cluster": "cluster"})
    df["pred"] = df["pred"].astype(str).str.strip()
    return df[["slide_id", "tile_id", "cluster", "pred"]].copy()


def write_tiles_csv_for_slide(
    slide_id: str,
    group: str,
    assign_csv: Path,
    h5_base: Path,
    out_dir: Path,
    tiles_csv_name: str,
) -> Path:
    df_assign = read_assign_csv(assign_csv)

    h5_path = h5_base / f"{slide_id}_features.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"[H5] Not found: {h5_path}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _, coords_np, tile_ids = get_features_from_h5(h5_path, device=device)
    id2bbox = build_id2bbox(coords_np, tile_ids)

    rows = []
    miss_tile = 0
    for _, r in df_assign.iterrows():
        tid = str(r["tile_id"]).strip()
        bbox = id2bbox.get(tid)
        if bbox is None:
            miss_tile += 1
            continue
        x, y, w, h = bbox
        rows.append(
            {
                "slide_id": str(r["slide_id"]),
                "tile_id": tid,
                "cluster": str(r["cluster"]),
                "pred": str(r["pred"]),
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            }
        )
    if miss_tile:
        logging.warning(f"[tiles] {slide_id}: missing {miss_tile} tile_id(s) in H5")

    out_dir = out_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / tiles_csv_name.format(slide_id=slide_id)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logging.info(f"[tiles] Wrote: {out_csv} (rows={len(rows)})")
    return out_csv


def find_assign_csv_for_slide(
    slide_id: str,
    mapping_out_dir: Path,
    patch_csv_pattern: str,
) -> Tuple[Path, str]:
    """
    Search under mapping_out_dir for {group}/{slide_id}_patch_assign.csv.
    Return (path, group).
    """
    pattern = f"**/{patch_csv_pattern.format(slide_id=slide_id)}"
    path = rglob_first(mapping_out_dir, pattern)
    if path is None:
        fallback = f"**/{slide_id}_patch_assign*.csv"
        path = rglob_first(mapping_out_dir, fallback)
        if path is None:
            raise FileNotFoundError(
                f"[assign] Not found under {mapping_out_dir} with patterns '{pattern}' or '{fallback}'"
            )
    group = path.parent.name  # expected: Clear / Endo / High / Border
    return path, group


# ---------------- GeoJSON creation (attention + spectrum) ----------------
def get_class_names_and_count(manifest: Path) -> Tuple[Dict[int, str], int]:
    df = pd.read_csv(manifest)
    labels = sorted(int(x) for x in df["label"].unique())
    class_names = {i: str(df[df["label"] == i]["class"].unique()[0]) for i in labels}
    return class_names, len(labels)


def process_one_slide_to_geojson(
    slide_path: Path,
    tiles_csv: Path,
    group: str,
    config: dict,
    h5_base: Path,
    attn_checkpoint: Path,
    manifest_path: Path,
    geojson_root: Path,
    device_str: str = "cuda",
    save_jpg: bool = False,
) -> Path:
    """
    Given a tiles CSV (spectrum assignments + coords) for one slide,
    run the attention model and export a single GeoJSON file
    combining attention tiles and spectrum-based annotations.
    """
    algo = config["algorithm"]
    out_geo = geojson_root / group / config["output"]["geojson_name"].format(slide_id=slide_path.stem)
    out_geo.parent.mkdir(parents=True, exist_ok=True)

    # Load spectrum tiles
    sp_df = pd.read_csv(tiles_csv)
    need = {"slide_id", "tile_id", "cluster", "x", "y", "width", "height", "pred"}
    miss = need - set(sp_df.columns)
    if miss:
        raise ValueError(f"[tiles] Missing columns {sorted(miss)} in {tiles_csv}")
    sp_df["slide_id"] = sp_df["slide_id"].astype(str).str.strip().str.replace(r"\.svs$", "", regex=True)
    sp_df["pred"] = sp_df["pred"].astype(str).str.strip()
    sp_df = sp_df[sp_df["slide_id"] == slide_path.stem].copy()

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cuda")

    # Open WSI and thumbnail for optional JPG
    wsi = openslide.open_slide(str(slide_path))
    display_image, scale_factor = get_display_image(wsi, int(algo.get("display_level", 2)))

    # H5 features for attention model
    h5_path = h5_base / f"{slide_path.stem}_features.h5"
    features, coords_np, _ = get_features_from_h5(h5_path, device=device)
    rects_scaled = scale_rectangles(coords_np, scale_factor)

    # Attention model & class manifest
    #class_names, n_classes = get_class_names_and_count(manifest_path)
    label_map_short = config["LABEL_MAP_SHORT"]
    class_names = {int(k): str(v) for k, v in label_map_short.items()}
    n_classes = int(config["N_CLASSES"])
    model = load_trained_model(
        device=device,
        checkpoint_path=attn_checkpoint,
        model_size=str(algo.get("attn_model_size", "small")),
        input_feature_size=int(algo.get("input_feature_size", 512)),
        n_classes=n_classes,
    )
    A_raw_np, prob_np = predict_all_attentions(model, features)  # [C, N], [C]

    # Palette: use config if present, otherwise a fixed fallback
    palette = [[255, 0, 0], [0, 153, 255], [0, 204, 102], [255, 165, 0], [128, 0, 128]]

    pred_entries: List[dict] = []
    features_out: List[dict] = []

    for cls_idx in range(n_classes):
        cls_name = class_names.get(cls_idx, f"class_{cls_idx}")
        cls_prob = float(prob_np[cls_idx])
        cls_prob_str = f"{cls_prob:.2f}"

        # Attention tiles
        attn_scores_cls = standardize_scores(A_raw_np[cls_idx].squeeze())
        for rect_bounds, s in zip(coords_np, attn_scores_cls):
            features_out.append(make_attention_tile_feature(tuple(rect_bounds), s, slide_path.stem, cls_name))

        # Optional JPG overlay
        if save_jpg:
            import cv2  # local import to keep dependency optional

            h, w = np.asarray(display_image).shape[:2]
            score_map = np.zeros((h, w), dtype=np.float32)
            for rect, score in zip(rects_scaled, attn_scores_cls):
                minx, miny, maxx, maxy = rect.bounds
                score_map[round(miny): round(maxy), round(minx): round(maxx)] = float(score)

            heatmap_bgr = cv2.applyColorMap(np.uint8(255 * score_map), cv2.COLORMAP_JET)
            heatmap = Image.fromarray(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGBA), mode="RGBA")
            heatmap.putalpha(60)
            result = Image.alpha_composite(display_image, heatmap)
            jpg_dir = out_geo.parent
            jpg_dir.mkdir(parents=True, exist_ok=True)
            out_jpg = jpg_dir / f"{slide_path.stem}_attn_PRED_{cls_name}.jpg"
            result.convert("RGB").save(str(out_jpg))

        # Spectrum annotations for THIS class
        # long_to_short = config["subtypes"]["long_to_short"]
        # if cls_name not in long_to_short:
        #     logging.warning(f"[class] {cls_name} not found in subtypes.long_to_short, skip spectrum anno for this class.")
        #     sp_cls = sp_df.iloc[0:0].copy()
        # else:
        #     short_code = long_to_short[cls_name]
        sp_cls = sp_df.loc[sp_df["pred"] == cls_name].copy()
        spec_count = 0
        for _, r in sp_cls.iterrows():
            feat = make_spectrum_annotation_feature(r, palette, cls_name, expect_slide_id=slide_path.stem)
            if feat is not None:
                features_out.append(feat)
                spec_count += 1

        cls_sets = build_classificationsets_from_df(sp_cls, CLASSIFICATION_SETS)
        pred_entries.append(
            {
                "prediction": cls_name,
                "confidence": cls_prob_str,
                "classificationsets": cls_sets,
            }
        )
        logging.info(
            f"[class] {cls_name}: prob={cls_prob_str}, tiles={len(attn_scores_cls)}, "
            f"spectrum_annots={spec_count}, sets={len(cls_sets)}"
        )

    export_multi_pred_geojson(pred_entries, features_out, out_geo)
    return out_geo


# ---------------- Orchestrator (one slide: assign → tiles → geojson) ----------------
def run_single_slide_explanation(
    config: dict,
    slide_path: Path,
    mapping_root: Path,
    h5_base: Path,
    tiles_root: Path,
    geojson_root: Path,
    attn_checkpoint: Path,
    manifest_path: Path,
    device: str = "cuda",
    save_jpg: bool = False,
) -> Dict[str, Path]:
    """
    Top-level function for Step 3 (single slide).

    Given:
      - a slide path (.svs),
      - spectrum-mapping outputs (patch_assign CSVs under mapping_root),
      - feature bags (H5 under h5_base),
      - output dirs for tiles and GeoJSON,
      - the full-data AttentionNet checkpoint,
      - a class manifest CSV,

    this function will:
      1) locate the patch assignment file for the slide,
      2) create a tiles CSV with spectrum cluster + coordinates,
      3) run the attention model and export a combined GeoJSON.

    Returns:
      {
        "tiles_csv": <Path to tiles CSV>,
        "geojson":   <Path to GeoJSON>,
      }
    """
    slide_path = slide_path.resolve()
    slide_id = infer_slide_id(slide_path)

    mapping_root = mapping_root.resolve()
    h5_base = h5_base.resolve()
    tiles_root = tiles_root.resolve()
    geojson_root = geojson_root.resolve()
    attn_checkpoint = attn_checkpoint.resolve()
    manifest_path = manifest_path.resolve()

    patch_pat = config["output"].get("patch_csv_pattern", "{slide_id}_patch_assign.csv")
    tiles_name = config["output"].get("tiles_csv_name", "{slide_id}_spectrum_tiles.csv")

    # 1) find patch assignment CSV
    assign_csv, group = find_assign_csv_for_slide(slide_id, mapping_root, patch_pat)
    logging.info(f"[assign] Found: {assign_csv} (group={group})")
    print(f"[assign] Found: {assign_csv} (group={group})")

    # 2) write tiles CSV (join coords from H5)
    tiles_csv = write_tiles_csv_for_slide(
        slide_id=slide_id,
        group=group,
        assign_csv=assign_csv,
        h5_base=h5_base,
        out_dir=tiles_root,
        tiles_csv_name=tiles_name,
    )

    # 3) generate GeoJSON (attention + spectrum)
    out_geo = process_one_slide_to_geojson(
        slide_path=slide_path,
        tiles_csv=tiles_csv,
        group=group,
        config=config,
        h5_base=h5_base,
        attn_checkpoint=attn_checkpoint,
        manifest_path=manifest_path,
        geojson_root=geojson_root,
        device_str=device,
        save_jpg=save_jpg,
    )

    logging.info(f"[DONE] Slide '{slide_id}' completed. GeoJSON: {out_geo}")
    return {"tiles_csv": tiles_csv, "geojson": out_geo}
