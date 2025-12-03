# spectrum_mapping.py
#
# Stage 2 – Morphologic Spectrum Mapping
#
# This module maps high-contribution patches from independent test slides
# onto a pre-computed morphologic spectrum.


from __future__ import annotations

import os
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ----------------------------- Utils ----------------------------- #

def parse_embed_vec(s: str) -> np.ndarray:
    """Parse a semicolon-separated embedding string into a float32 vector."""
    return np.array([float(x) for x in str(s).split(";") if x != ""], dtype=np.float32)


def l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2D array."""
    norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norm


def _rbf_weights(D: np.ndarray, tau_vec: np.ndarray) -> np.ndarray:
    """RBF-based soft assignment weights for a distance matrix D."""
    tau2 = (np.maximum(tau_vec, 1e-6) ** 2)[None, :]
    Z = - (D ** 2) / (2.0 * tau2)
    Z = Z - Z.max(axis=1, keepdims=True)
    Wt = np.exp(Z)
    W = Wt / (Wt.sum(axis=1, keepdims=True) + 1e-12)
    return W


def _knn_search(query: np.ndarray, bank: np.ndarray, k: int, metric: str = "cosine"):
    """Return indices and distances of kNN for a single query vector."""
    k_eff = min(k, bank.shape[0])
    if k_eff == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    if metric == "cosine":
        sims = bank @ query
        idx = np.argpartition(-sims, kth=k_eff - 1)[:k_eff]
        idx = idx[np.argsort(-sims[idx])]
        sims_k = sims[idx]
        dist = 1.0 - sims_k
        return idx, dist
    else:
        diff = bank - query[None, :]
        d2 = np.sum(diff * diff, axis=1)
        idx = np.argpartition(d2, kth=k_eff - 1)[:k_eff]
        idx = idx[np.argsort(d2[idx])]
        dist = np.sqrt(d2[idx])
        return idx, dist


# ----------------------------- IO: test embeddings ----------------------------- #

def read_test_embeddings(
    test_embeds: Dict[str, Path],
    embed_col: str,
    coord_cols: List[str],
) -> pd.DataFrame:
    """
    Read test embeddings from either:
      1) a single unified CSV (Path/str), or
      2) a dict of {name: CSV_path}.

    Each CSV must include:
      - slide_id
      - tile_id
      - embed_col (semicolon-separated embeddings)
      - pred      (short prediction code, e.g. Clear/Endo/High/Border)

    Optional:
      - label     (true short label, used only as fallback)
      - x, y, w, h
    """

    
    dfs = []
    for name, fp in test_embeds.items():
        fp_str = str(fp)
        if not os.path.exists(fp_str):
            logging.warning(f"[test] Embedding file not found: {fp_str} (skip)")
            continue
        df = pd.read_csv(fp_str)
        need = {"slide_id", "tile_id", embed_col, "pred"}
        if not need.issubset(df.columns):
            missing = sorted(need - set(df.columns))
            raise ValueError(f"[test] {fp_str} missing columns: {missing}")
        emb = np.stack(df[embed_col].apply(parse_embed_vec).values)
        emb = l2_normalize_rows(emb)

        keep = ["slide_id", "tile_id", "pred", "label"] + [c for c in coord_cols if c in df.columns]
        keep = [c for c in keep if c in df.columns]
        out = df[keep].copy()
        out["embed"] = list(emb)
        dfs.append(out)
        logging.info(f"[test] Loaded embeddings: {name} ({len(out):,} patches)")

    if not dfs:
        raise ValueError("No test embeddings were loaded.")
    return pd.concat(dfs, ignore_index=True)


# ----------------------------- each_slide_result ----------------------------- #

def _strict_subtype_series(s: pd.Series, mapping: Dict[str, str], colname: str) -> pd.Series:
    s_clean = s.astype(str).str.strip()
    unmapped = sorted(set(s_clean.unique()) - set(mapping.keys()))
    if unmapped:
        raise ValueError(
            f"[each_slide] Column '{colname}' contains unknown labels: {unmapped}\n"
            f"Allowed values: {list(mapping.keys())}"
        )
    return s_clean.map(mapping)


def load_true_pred_table(each_slide_csv: str | Path, mapping: Dict[str, str]):
    """
    Load slide-level full-data model results.

    each_slide_csv must contain:
      - slide_id
      - name_label   (long-form ground-truth subtype)
      - name_pred    (long-form predicted subtype)
      - correctness  (True/False or equivalent)
    """
    df = pd.read_csv(each_slide_csv)
    df["true_short"] = _strict_subtype_series(df["name_label"], mapping, "name_label")
    df["pred_short"] = _strict_subtype_series(df["name_pred"], mapping, "name_pred")
    truthy = {"true", "1", "yes", "y", "t"}
    df["correct"] = df["correctness"].astype(str).str.strip().str.lower().isin(truthy)
    slide2true = dict(zip(df["slide_id"].astype(str), df["true_short"]))
    slide2pred = dict(zip(df["slide_id"].astype(str), df["pred_short"]))
    slide2corr = dict(zip(df["slide_id"].astype(str), df["correct"]))
    return slide2true, slide2pred, slide2corr


# ----------------------------- Assignment ----------------------------- #

def soft_assign_one_slide(
    df_slide: pd.DataFrame,
    stats: dict,
    cluster_list: List[str],
    knn_bank: dict,
    slide_group: str,
    algo_cfg: dict,
) -> pd.DataFrame:
    """
    Soft-assign the patches of a SINGLE-PRED subset of one slide.

    slide_group:
      "Endo_High"/"High_Endo" to enable hybrid EC↔HGSC rule;
      otherwise a true short label ("Clear","Endo","High","Border").
    """
    X = np.stack(df_slide["embed"].values)
    N, C = X.shape[0], len(cluster_list)

    Dmat = np.zeros((N, C), dtype=np.float32)
    tau_vec = np.zeros((C,), dtype=np.float32)
    r95_vec = np.zeros((C,), dtype=np.float32)

    for j, cname in enumerate(cluster_list):
        st = stats[cname]
        mu = st["mu"]
        invS = st["invS"]
        diff = X - mu[None, :]
        Dmat[:, j] = np.sqrt(np.sum(diff @ invS * diff, axis=1))
        tau_vec[j] = st["tau"]
        r95_vec[j] = st["r95"]

    d_min = Dmat.min(axis=1)
    near_idx = Dmat.argmin(axis=1)
    nearest_cluster = np.array(cluster_list)[near_idx]
    nearest_r95 = r95_vec[near_idx]

    W = _rbf_weights(Dmat, tau_vec)
    outlier = (d_min > nearest_r95)
    assigned = np.where(outlier, "Outlier", nearest_cluster)

    # Hybrid only for EC↔HGSC
    use_mixture = slide_group in {"Endo_High", "High_Endo"}

    knn_k = int(algo_cfg.get("knn_k", 30))
    knn_distance = str(algo_cfg.get("knn_distance", "cosine"))
    ptrans_abs_min = float(algo_cfg.get("ptrans_abs_min", 0.04))
    conf_min = float(algo_cfg.get("conf_min", 0.55))
    imbalance_alpha = float(algo_cfg.get("imbalance_alpha", 0.5))
    relax_rel = float(algo_cfg.get("outlier_relax_rel", 0.10))
    relax_abs = float(algo_cfg.get("outlier_relax_abs", 4.0))

    if use_mixture:
        # Relaxed outlier threshold for EC↔HGSC mixed cases
        relax_thr = np.maximum(nearest_r95 * (1.0 + relax_rel), nearest_r95 + relax_abs)
        outlier_relaxed = (d_min > relax_thr)
        outlier = outlier_relaxed
        assigned = np.where(outlier, "Outlier", nearest_cluster)

        is_core_EC = np.array([str(c).startswith("Endo:") for c in nearest_cluster], dtype=bool)
        is_core_HG = np.array([str(c).startswith("High:") for c in nearest_cluster], dtype=bool)
        candidate = (~outlier) & (is_core_EC | is_core_HG)

        emb_bank = knn_bank["emb"]
        clus_bank = knn_bank["clus"]
        is_trans_bank = knn_bank["is_trans"]
        trans_counts = knn_bank["trans_counts"]

        ptrans_arr = np.zeros(N, dtype=np.float32)
        conf_arr = np.zeros(N, dtype=np.float32)
        is_low_conf = np.zeros(N, dtype=np.int32)

        for i in range(N):
            if not candidate[i]:
                continue
            q = X[i, :]
            idx_k, dist_k = _knn_search(q, emb_bank, k=knn_k, metric=knn_distance)
            if idx_k.size == 0:
                continue

            w = 1.0 / (dist_k + 1e-3)
            is_trans = is_trans_bank[idx_k]
            w_all = w.sum() + 1e-12
            w_trans = w[is_trans].sum()
            p_trans = float(w_trans / w_all)
            ptrans_arr[i] = p_trans

            if p_trans < ptrans_abs_min:
                continue

            idx_trans = idx_k[is_trans]
            w_trans_vec = w[is_trans]
            clus_trans = clus_bank[idx_trans]

            score: Dict[str, float] = {}
            for cj, wj in zip(clus_trans, w_trans_vec):
                nt = float(trans_counts.get(str(cj), 1.0))
                score[cj] = score.get(cj, 0.0) + (wj / (nt ** imbalance_alpha))

            if not score:
                continue
            items = sorted(score.items(), key=lambda kv: -kv[1])
            best_c, best_s = items[0]
            sec_s = items[1][1] if len(items) > 1 else 0.0
            conf = float(best_s / (best_s + sec_s + 1e-12))

            conf_arr[i] = conf
            is_low_conf[i] = int(conf < conf_min)
            assigned[i] = best_c

    # Output per-pred subset; ensure 'pred' is preserved from df_slide
    keep = ["slide_id", "tile_id", "pred"] + [c for c in ["x", "y", "w", "h"] if c in df_slide.columns]
    keep = [c for c in keep if c in df_slide.columns]
    df_out = df_slide[keep].copy()
    df_out["assigned_cluster"] = assigned
    df_out["min_d"] = d_min
    df_out["nearest_cluster"] = nearest_cluster
    df_out["nearest_r95"] = nearest_r95
    df_out["is_outlier"] = outlier.astype(int)

    # Top-2 diagnostics
    idx_sort_byD = np.argsort(Dmat, axis=1)
    top1_idx = idx_sort_byD[:, 0]
    top2_idx = idx_sort_byD[:, 1] if Dmat.shape[1] >= 2 else top1_idx
    df_out["top2_cluster"] = np.array(cluster_list)[top2_idx]
    df_out["top2_dist"] = Dmat[np.arange(N), top2_idx]
    df_out["top2_r95"] = r95_vec[top2_idx]

    W = _rbf_weights(Dmat, tau_vec)
    df_out["top1_w"] = W[np.arange(N), top1_idx]
    df_out["top2_w"] = W[np.arange(N), top2_idx]

    return df_out


# ----------------------------- Top-level API ----------------------------- #

def run_spectrum_mapping(
    config: dict,
    spectrum_stats_path: Path,
    test_embeds: Dict[str, Path],
    each_slide_result: Path,
    output_dir: Path,
    embed_col: str = "embed",
    coord_cols: List[str] = ("x", "y", "w", "h"),
) -> Path:
    """
    Run Stage 2 spectrum mapping.

    Args
    ----
    config:
        Full config dict (loaded by the caller). Only 'subtypes', 'algorithm',
        and 'output' sections are required.
    spectrum_stats_path:
        Path to morphologic_spectrum_stats.pkl exported by Stage 1.
    test_embeds:
        Either a single unified CSV path, or a dict of {name: CSV_path}.
    each_slide_result:
        CSV with slide-level results (slide_id, name_label, name_pred, correctness).
    output_dir:
        Directory where patch-level assignment CSVs and slide×cluster matrix
        will be written.
    embed_col:
        Column name for embeddings in test_embeds CSV(s).
    coord_cols:
        Coordinate columns to carry through if present.

    Returns
    -------
    matrix_path : Path
        Path to the slide×cluster matrix CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # A) Load spectrum stats
    with open(spectrum_stats_path, "rb") as f:
        payload = pickle.load(f)
    stats = payload["stats"]
    cluster_list = payload["cluster_list"]
    knn_bank = payload["knn_bank"]

    # B) Load test embeddings
    df_test = read_test_embeddings(test_embeds, embed_col, list(coord_cols))

    # C) Slide-level truth & prediction
    slide2true, slide2pred1, slide2corr = load_true_pred_table(
        each_slide_result,
        config["subtypes"]["long_to_short"],
    )

    patch_csv_pattern = config["output"].get("patch_csv_pattern", "{slide_id}_patch_assign.csv")
    matrix_name = config["output"].get("matrix_name", "slide_cluster_matrix.csv")

    algo_cfg = config["algorithm"]
    min_patches_per_slide = int(algo_cfg.get("min_patches_per_slide", 10))

    matrix_rows: List[dict] = []
    n_written = 0

    for sid, sub_all in df_test.groupby("slide_id"):
        if len(sub_all) < min_patches_per_slide:
            logging.warning(f"[warn] {sid}: total patches={len(sub_all)} < min_patches_per_slide")

        true_short: Optional[str] = slide2true.get(str(sid))
        if true_short is None:
            if "label" in sub_all.columns and len(sub_all["label"].dropna()) > 0:
                true_short = str(sub_all["label"].iloc[0]).strip()
                logging.warning(f"[fallback] {sid}: using label='{true_short}' from test CSV")
            else:
                logging.warning(f"[skip] {sid}: not found in each_slide_result and no 'label' column – skip")
                continue

        per_pred_out: List[pd.DataFrame] = []
        for pred_name, sub_pred in sub_all.groupby("pred"):
            sub_pred = sub_pred.reset_index(drop=True)
            if len(sub_pred) == 0:
                continue

            if {true_short, str(pred_name)} == {"Endo", "High"}:
                slide_group = "Endo_High" if true_short == "Endo" else "High_Endo"
            else:
                slide_group = true_short

            df_out_one_pred = soft_assign_one_slide(
                df_slide=sub_pred,
                stats=stats,
                cluster_list=cluster_list,
                knn_bank=knn_bank,
                slide_group=slide_group,
                algo_cfg=algo_cfg,
            )
            if "pred" not in df_out_one_pred.columns:
                df_out_one_pred["pred"] = pred_name
            per_pred_out.append(df_out_one_pred)

        if not per_pred_out:
            logging.warning(f"[skip] {sid}: no valid pred subset outputs")
            continue

        df_merged = pd.concat(per_pred_out, ignore_index=True)

        group_dir = output_dir / true_short
        group_dir.mkdir(parents=True, exist_ok=True)
        patch_fp = group_dir / patch_csv_pattern.format(slide_id=sid)
        df_merged.to_csv(patch_fp, index=False)
        logging.info(f"[OK] Wrote patch-level assignment: {patch_fp} (rows={len(df_merged)})")
        n_written += 1

        counts = df_merged["assigned_cluster"].astype(str).value_counts(normalize=True)
        row = {"slide_id": sid, "group": true_short}
        for c in cluster_list:
            row[c] = float(counts.get(c, 0.0))
        row["Outlier"] = float(counts.get("Outlier", 0.0))
        matrix_rows.append(row)

    matrix_path = output_dir / matrix_name
    if n_written == 0:
        logging.warning("[WARN] No slides were written – please check test_embeds / each_slide_result / inputs")
        return matrix_path

    if matrix_rows:
        col_order = ["slide_id", "group"] + cluster_list + ["Outlier"]
        mat_wide_df = pd.DataFrame(matrix_rows, columns=col_order)
        mat_wide_df.to_csv(matrix_path, index=False)
        logging.info(f"[OK] Wrote slide×cluster matrix: {matrix_path}")

    return matrix_path
