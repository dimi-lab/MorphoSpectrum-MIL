#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-slide pipeline:
1) Read patch assignment CSV (from spectrum_mapping.py outputs), join H5 coords → write tiles CSV
2) Run attention model once, combine attention + spectrum annotations → write ONE GeoJSON

Usage:
  python spectrum_tiles_and_geojson.py --config config.yaml --input-slide /abs/path/XXX.svs \
      [--id-match exact|endswith|contains] [--save-jpg]

Notes:
- Default does NOT save per-class JPG; pass --save-jpg to enable QA jpg overlays.
- We assume current working directory is ./independent_explain and the model.py resides in ../interpretability.
"""

import argparse
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
import yaml
from PIL import Image
from shapely.affinity import scale as shp_scale
from shapely.geometry import box

# ---------------- Model import (project layout aware) ----------------
# Current script expected under ./independent_explain ; model.py under ./interpretability
_THIS_DIR = Path(__file__).resolve().parent
_PROJEndoT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJEndoT_ROOT / "interpretability"))
try:
    from model import AttentionNet
except Exception as e:
    raise ImportError(
        "Failed to import AttentionNet from ../interpretability/model.py. "
        "Please ensure the repo layout is:\n"
        "  ./interpretability/model.py\n"
        "  ./independent_explain/spectrum_tiles_and_geojson.py\n"
        f"Import error: {e}"
    )

# ---------------- Pretty logging ----------------
def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------- Config ----------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ["paths", "subtypes", "algorithm", "output"]:
        if key not in cfg:
            raise ValueError(f"Missing top-level section '{key}' in config.yaml")
    return cfg

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
    with h5py.File(h5_path, "r") as f:
        feats = f["features"][:]
        tile_ids_all = f["tile_ids"][:]
        coords_all = f["coords"][:]
    tile_ids_str = [tid.decode("utf-8") if isinstance(tid, (bytes, bytearray)) else str(tid) for tid in tile_ids_all]

    tissue_feats, cluster_feats, cell_feats = [], [], []
    tissue_coords, cluster_coords, cell_coords = [], [], []
    tissue_ids, cluster_ids, cell_ids = [], [], []

    for feature, coord, tid in zip(feats, coords_all, tile_ids_str):
        if tid.startswith("tissue"):
            tissue_feats.append(feature); tissue_coords.append(coord); tissue_ids.append(tid)
        elif tid.startswith("cluster"):
            cluster_feats.append(feature); cluster_coords.append(coord); cluster_ids.append(tid)
        elif tid.startswith("cell"):
            cell_feats.append(feature); cell_coords.append(coord); cell_ids.append(tid)
        else:
            # tolerate: default to cluster scale
            cluster_feats.append(feature); cluster_coords.append(coord); cluster_ids.append(tid)

    to_np = lambda x: np.array(x, dtype=np.float32)
    return {
        "tissue":  (to_np(tissue_feats),  np.array(tissue_coords), tissue_ids),
        "cluster": (to_np(cluster_feats), np.array(cluster_coords), cluster_ids),
        "cell":    (to_np(cell_feats),    np.array(cell_coords),   cell_ids),
    }

def get_features_from_h5(h5_path: Path, scale: str, device: torch.device):
    bags = split_feature_tile_id(h5_path)
    if scale not in bags:
        raise ValueError(f"[H5] scale '{scale}' not in {list(bags.keys())} for {h5_path}")
    feats_np, coords_np, tile_ids = bags[scale]
    if feats_np.size == 0:
        raise ValueError(f"[H5] empty bag for scale '{scale}' in {h5_path}")
    features = torch.from_numpy(feats_np).to(device)
    return features, coords_np, tile_ids

# ---------------- Matching helpers ----------------
def build_id2bbox(coords: np.ndarray, ids: List[str]) -> Dict[str, Tuple[int, int, int, int]]:
    id2bbox = {}
    for c, tid in zip(coords, ids):
        minx, miny, maxx, maxy = [int(v) for v in c]
        id2bbox[tid] = (minx, miny, maxx - minx, maxy - miny)
    return id2bbox

def find_bbox_for_tile(tile_id: str, id2bbox: Dict[str, Tuple[int, int, int, int]], scale: str, match: str) -> Optional[Tuple[int, int, int, int]]:
    if tile_id in id2bbox:
        return id2bbox[tile_id]
    prefixed = f"{scale}_{tile_id}"
    if prefixed in id2bbox:
        return id2bbox[prefixed]
    if match == "endswith":
        for k in id2bbox.keys():
            if k.endswith(tile_id):
                return id2bbox[k]
    elif match == "contains":
        for k in id2bbox.keys():
            if tile_id in k:
                return id2bbox[k]
    return None

# ---------------- Attention model ----------------
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

def predict_all_attentions(model, feature_batch: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits, Y_prob, Y_hat, A_raw, *_ = model(feature_batch)
    A_raw_np = A_raw.detach().cpu().numpy()              # [C, N]
    if Y_prob is not None:
        prob_np = Y_prob.detach().cpu().view(-1).numpy() # [C]
    else:
        lg = logits.detach().cpu().view(-1).numpy()
        ex = np.exp(lg - np.max(lg)); prob_np = ex / np.sum(ex)
    return A_raw_np, prob_np

# ---------------- Spectrum labels (human-readable) ----------------
CLASSIFICATION_SETS = {
    # (same mapping as your existing script; edit/extend as needed)
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
        groups.append(g); hc_names.append(hc); full_keys.append(full)
    uniq_full = sorted(set(full_keys),
                       key=lambda key: int(re.search(r'\d+', key).group()) if re.search(r'\d+', key) else 10**9)
    out = []
    for fk in uniq_full:
        name_fmt = fk.replace(":", "_")
        descr = class_desc_dict.get(fk, "unknown")
        out.append({"name": name_fmt, "descr": descr})
    if not uniq_full and hc_names:
        uniq_hc = sorted(set(hc_names),
                         key=lambda h: int(re.search(r'\d+', h).group()) if re.search(r'\d+', h) else 10**9)
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
            "metadata": {"slide_id": slide_id, "coord_unit": "pixel", "wsi_level": 0}
        }
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

    # 按“簇”决定颜色
    hc_name = extract_hc_name(cluster)  # e.g. "Endo_hc_4"
    color = palette[hc_to_color_index(hc_name, len(palette))]
    ms_idx = hc_index(hc_name)

    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": rect_to_polygon(x, y, w, h)},
        "properties": {
            "prediction": pred_class_name,      # 该要素属于哪个 prediction（类显示名）
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
        }
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
    # normalize pred to Clear/Endo/High/Border 
    df["pred"] = df["pred"].astype(str).str.strip()
    bad = set(df["pred"].unique()) - {"Clear", "Endo", "High", "Border"}
    if bad:
        raise ValueError(f"[assign] Unexpected pred codes {sorted(bad)} in {assign_csv}; expected Clear/Endo/High/Border.")
    # keep minimal cols; if coords already present, keep them but they will be overwritten by H5 match
    keep = ["slide_id", "tile_id", "cluster", "pred"]
    for c in ["x", "y", "width", "height"]:
        if c in df.columns:
            keep.append(c)
    return df[keep].copy()

def write_tiles_csv_for_slide(
    slide_id: str,
    group: str,
    assign_csv: Path,
    h5_base: Path,
    scale: str,
    out_dir: Path,
    tiles_csv_name: str,
    id_match: str = "exact"
) -> Path:
    df_assign = read_assign_csv(assign_csv)
    # H5 locate
    h5_path = h5_base / f"{slide_id}_features.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"[H5] Not found: {h5_path}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _, coords_np, tile_ids = get_features_from_h5(h5_path, scale=scale, device=device)
    id2bbox = build_id2bbox(coords_np, tile_ids)

    rows = []
    miss_tile = 0
    for _, r in df_assign.iterrows():
        tid = str(r["tile_id"])
        bbox = find_bbox_for_tile(tid, id2bbox, scale=scale, match=id_match)
        if bbox is None:
            miss_tile += 1
            continue
        x, y, w, h = bbox
        rows.append({
            "slide_id": str(r["slide_id"]),
            "tile_id": tid,
            "cluster": str(r["cluster"]),
            "pred": str(r["pred"]),
            "x": x, "y": y, "width": w, "height": h
        })
    if miss_tile:
        logging.warning(f"[tiles] {slide_id}: missing {miss_tile} tile_id(s) in H5 (match='{id_match}')")

    out_dir = out_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / tiles_csv_name.format(slide_id=slide_id)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logging.info(f"[tiles] Wrote: {out_csv} (rows={len(rows)})")
    return out_csv

def find_assign_csv_for_slide(slide_id: str, mapping_out_dir: Path, patch_csv_pattern: str) -> Tuple[Path, str]:
    """
    Search under mapping output root for {group}/{slide_id}_patch_assign*.csv.
    Return (path, group).
    """
    pattern = f"**/{patch_csv_pattern.format(slide_id=slide_id)}"
    # allow wildcard variants like *_patch_assign*.csv if pattern includes *
    path = rglob_first(mapping_out_dir, pattern)
    if path is None:
        # try common fallback
        fallback = f"**/{slide_id}_patch_assign*.csv"
        path = rglob_first(mapping_out_dir, fallback)
        if path is None:
            raise FileNotFoundError(f"[assign] Not found under {mapping_out_dir} with patterns '{pattern}' or '{fallback}'")
    group = path.parent.name  # expected to be Clear/Endo/High/Border
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
    cfg: dict,
    save_jpg: bool,
    id_match: str
):
    paths = cfg["paths"]
    algo = cfg["algorithm"]
    out_geo_root = Path(paths["geojson_root"])
    out_geo = out_geo_root / group / cfg["output"]["geojson_name"].format(slide_id=slide_path.stem)
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

    # Open slide + H5 features (for attention tiles)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    wsi = openslide.open_slide(str(slide_path))
    display_image, scale_factor = get_display_image(wsi, int(algo.get("display_level", 2)))

    # H5 features for attention model
    h5_path = Path(paths["h5_base"]) / f"{slide_path.stem}_features.h5"
    features, coords_np, _ = get_features_from_h5(h5_path, scale=str(algo.get("h5_scale", "cluster")), device=device)
    rects_scaled = scale_rectangles(coords_np, scale_factor)

    # Attention model & manifest
    class_names, n_classes = get_class_names_and_count(Path(paths["manifest"]))
    model = load_trained_model(
        device=device,
        checkpoint_path=str(paths["attn_checkpoint"]),
        model_size=str(algo.get("attn_model_size", "small")),
        input_feature_size=int(algo.get("input_feature_size", 512)),
        n_classes=n_classes,
    )
    A_raw_np, prob_np = predict_all_attentions(model, features)  # [C,N], [C]

    palette = [ [255,0,0], [0,153,255], [0,204,102], [255,165,0], [128,0,128] ]  # red, blue, green, orange, purple
    pred_entries: List[dict] = []
    features_out: List[dict] = []

    # Build attention tiles once per class
    for cls_idx in range(n_classes):
        cls_name = class_names.get(cls_idx, f"class_{cls_idx}")
        cls_prob = float(prob_np[cls_idx])
        cls_prob_str = f"{cls_prob:.2f}"

        # Standardize attentions and render tile features
        attn_scores_cls = standardize_scores(A_raw_np[cls_idx].squeeze())
        for rect_bounds, s in zip(coords_np, attn_scores_cls):
            features_out.append(make_attention_tile_feature(tuple(rect_bounds), s, slide_path.stem, cls_name))

        # Optional JPG overlay
        if save_jpg:
            # Build score map at display resolution
            h, w = np.asarray(display_image).shape[:2]
            score_map = np.zeros((h, w), dtype=np.float32)
            for rect, score in zip(rects_scaled, attn_scores_cls):
                minx, miny, maxx, maxy = rect.bounds
                score_map[round(miny): round(maxy), round(minx): round(maxx)] = float(score)
            import cv2  # local import to avoid hard dependency if not saving JPGs
            heatmap_bgr = cv2.applyColorMap(np.uint8(255 * score_map), cv2.COLORMAP_JET)
            heatmap = Image.fromarray(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGBA), mode="RGBA")
            heatmap.putalpha(60)
            result = Image.alpha_composite(display_image, heatmap)
            jpg_dir = out_geo.parent
            jpg_dir.mkdir(parents=True, exist_ok=True)
            out_jpg = jpg_dir / f"{slide_path.stem}_attn_PRED_{cls_name}.jpg"
            result.convert("RGB").save(str(out_jpg))

        # Spectrum annotations for THIS class
        short_code = cfg["subtypes"]["long_to_short"][cls_name]  # Clear / Endo / High / Border
        sp_cls = sp_df.loc[sp_df["pred"] == short_code].copy()
        # Choose color by class index (stable)
        spec_count = 0
        for _, r in sp_cls.iterrows():
            feat = make_spectrum_annotation_feature(r, palette, cls_name, expect_slide_id=slide_path.stem)
            if feat is not None:
                features_out.append(feat)
                spec_count += 1

        # Classificationsets
        cls_sets = build_classificationsets_from_df(sp_cls, CLASSIFICATION_SETS)

        pred_entries.append({
            "prediction": cls_name,
            "confidence": cls_prob_str,
            "classificationsets": cls_sets
        })
        logging.info(f"[class] {cls_name}: prob={cls_prob_str}, tiles={len(attn_scores_cls)}, spectrum_annots={spec_count}, sets={len(cls_sets)}")

    export_multi_pred_geojson(pred_entries, features_out, out_geo)
    return out_geo

# ---------------- Orchestrator (one-slide, tiles → geojson) ----------------
def run_pipeline_single_slide(cfg_path: str, input_slide: Path, id_match: str, save_jpg: bool):
    cfg = load_config(cfg_path)
    slide_id = infer_slide_id(input_slide)

    # 1) Find assign CSV produced by spectrum_mapping.py
    mapping_root = Path(cfg["paths"]["output_dir"]).resolve()
    patch_pat = cfg["output"].get("patch_csv_pattern", "{slide_id}_patch_assign.csv")
    assign_csv, group = find_assign_csv_for_slide(slide_id, mapping_root, patch_pat)
    logging.info(f"[assign] Found: {assign_csv} (group={group})")

    # 2) Write tiles CSV (assign + H5 coords)
    tiles_root = Path(cfg["paths"]["spectrum_tiles_dir"]).resolve()
    tiles_name = cfg["output"].get("tiles_csv_name", "{slide_id}_spectrum_tiles.csv")
    tiles_csv = write_tiles_csv_for_slide(
        slide_id=slide_id,
        group=group,
        assign_csv=assign_csv,
        h5_base=Path(cfg["paths"]["h5_base"]).resolve(),
        scale=str(cfg["algorithm"].get("h5_scale", "cluster")),
        out_dir=tiles_root,
        tiles_csv_name=tiles_name,
        id_match=id_match,
    )

    # 3) Produce GeoJSON (attention + spectrum)
    out_geo = process_one_slide_to_geojson(
        slide_path=input_slide.resolve(),
        tiles_csv=tiles_csv,
        group=group,
        cfg=cfg,
        save_jpg=save_jpg,
        id_match=id_match,
    )
    logging.info(f"[DONE] Slide '{slide_id}' completed. GeoJSON: {out_geo}")

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Single-slide: tiles CSV (assign+coords) + GeoJSON (attention+spectrum)")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--input-slide", required=True, help="Path to one input WSI (.svs)")
    parser.add_argument("--id-match", default="exact", choices=["exact", "endswith", "contains"],
                        help="tile_id matching strategy when joining with H5 coords (default: exact)")
    parser.add_argument("--save-jpg", action="store_true", help="Save per-class JPG overlays (default: OFF)")
    args = parser.parse_args()

    setup_logging()
    input_slide = Path(args.input_slide)
    if not input_slide.exists():
        raise FileNotFoundError(f"input-slide not found: {input_slide}")

    run_pipeline_single_slide(cfg_path=args.config, input_slide=input_slide, id_match=args.id_match, save_jpg=args.save_jpg)

if __name__ == "__main__":
    main()
