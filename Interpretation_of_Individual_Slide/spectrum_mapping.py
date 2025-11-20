#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1 (publish-ready):
Consensus morphologic spectrum stats (incl. transitions) + per-pred soft assignment per slide,
then MERGE all preds of the same slide into ONE patch CSV (with a 'pred' column),
and (optionally) write a slide×cluster wide matrix aggregated over all preds.

Key behaviors aligned to your latest internal script:
  - Test inputs contain high-contribution patches for ALL predictions (column 'pred')
  - For a slide, we split by (slide_id, pred), run soft-assign once per pred subset,
    and concatenate the outputs → '{OUT_ROOT}/{true_short}/{slide_id}_patch_assign.csv'
  - Output root is grouped by TRUE short label (CC/EC/HGSC/SBT), derived from each_slide_result
  - EC↔HGSC pairs use the hybrid rule; others use Mahalanobis + r95

Configuration: minimal config.yaml with sections: paths / subtypes / algorithm / output
"""

import os
import sys
import glob
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ----------------------------- CLI & Config ----------------------------- #

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ["paths", "subtypes", "algorithm", "output"]:
        if key not in cfg:
            raise ValueError(f"Missing top-level section '{key}' in config.yaml")
    return cfg

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ----------------------------- Utils ----------------------------- #

def parse_embed_vec(s: str) -> np.ndarray:
    return np.array([float(x) for x in str(s).split(';') if x != ''], dtype=np.float32)

def l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norm

def _shrink_cov(S: np.ndarray, lam: float) -> np.ndarray:
    diag = np.diag(np.diag(S))
    return (1 - lam) * S + lam * diag

def _mahalanobis(X: np.ndarray, mu: np.ndarray, invS: np.ndarray) -> np.ndarray:
    diff = X - mu[None, :]
    return np.sqrt(np.sum(diff @ invS * diff, axis=1))

def _iqr(arr: np.ndarray) -> float:
    q75, q25 = np.percentile(arr, [75, 25])
    return float(max(q75 - q25, 1e-6))

def _rbf_weights(D: np.ndarray, tau_vec: np.ndarray) -> np.ndarray:
    tau2 = (np.maximum(tau_vec, 1e-6) ** 2)[None, :]
    Z = - (D ** 2) / (2.0 * tau2)
    Z = Z - Z.max(axis=1, keepdims=True)
    Wt = np.exp(Z)
    W = Wt / (Wt.sum(axis=1, keepdims=True) + 1e-12)
    return W

def _knn_search(query: np.ndarray, bank: np.ndarray, k: int, metric: str = "cosine"):
    k_eff = min(k, bank.shape[0])
    if k_eff == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)
    if metric == "cosine":
        sims = bank @ query
        idx = np.argpartition(-sims, kth=k_eff-1)[:k_eff]
        idx = idx[np.argsort(-sims[idx])]
        sims_k = sims[idx]
        dist = 1.0 - sims_k
        return idx, dist
    else:
        diff = bank - query[None, :]
        d2 = np.sum(diff * diff, axis=1)
        idx = np.argpartition(d2, kth=k_eff-1)[:k_eff]
        idx = idx[np.argsort(d2[idx])]
        dist = np.sqrt(d2[idx])
        return idx, dist

# ----------------------------- IO (Training) ----------------------------- #

def read_train_embeddings(paths_cfg: dict, embed_col: str) -> pd.DataFrame:
    dfs = []
    train_embeds: Dict[str, str] = paths_cfg.get("train_embeds", {}) or {}
    if not train_embeds:
        raise ValueError("paths.train_embeds is empty in config.")

    for parent, fp in train_embeds.items():
        if not os.path.exists(fp):
            logging.warning(f"[train] Embedding file not found: {fp} (skip)")
            continue
        df = pd.read_csv(fp)
        need = {"slide_id", "tile_id", embed_col}
        if not need.issubset(df.columns):
            missing = need - set(df.columns)
            raise ValueError(f"[train] {fp} missing columns: {missing}")
        emb = np.stack(df[embed_col].apply(parse_embed_vec).values)
        emb = l2_normalize_rows(emb)
        out = df[["slide_id", "tile_id"]].copy()
        out["embed"] = list(emb)
        out["parent"] = parent  # Clear / Endo / High / Border
        dfs.append(out)
        logging.info(f"[train] Loaded core embeddings: {parent} ({len(out):,} patches)")

    trans_embed_map: Dict[str, str] = paths_cfg.get("transition_embeds", {}) or {}
    for parent, fp in trans_embed_map.items():
        if not os.path.exists(fp):
            logging.warning(f"[train] Transition embedding file not found: {fp} (skip)")
            continue
        df = pd.read_csv(fp)
        need = {"slide_id", "tile_id", embed_col}
        if not need.issubset(df.columns):
            missing = need - set(df.columns)
            raise ValueError(f"[train] {fp} missing columns: {missing}")
        emb = np.stack(df[embed_col].apply(parse_embed_vec).values)
        emb = l2_normalize_rows(emb)
        out = df[["slide_id", "tile_id"]].copy()
        out["embed"] = list(emb)
        out["parent"] = parent  # Endo_to_High / High_to_Endo
        dfs.append(out)
        logging.info(f"[train] Loaded transition embeddings: {parent} ({len(out):,} patches)")

    if not dfs:
        raise ValueError("No training embeddings were loaded.")
    return pd.concat(dfs, ignore_index=True)

def read_cluster_dir_fixed(dirpath: str, name_prefix: str) -> pd.DataFrame:
    recs = []
    for fp in glob.glob(os.path.join(dirpath, "*.csv")):
        df = pd.read_csv(fp, header=None)
        if df.shape[1] < 7:
            raise ValueError(f"[train_clusters] {fp} has <7 columns")
        slide_ids = df.iloc[:, 0].astype(str).str.replace(".svs", "", regex=False)
        tile_ids  = df.iloc[:, 1].astype(str)
        hc_raw    = df.iloc[:, 2].astype(str).str.strip().str.lower()
        x = df.iloc[:, 3].astype(float)
        y = df.iloc[:, 4].astype(float)
        w = df.iloc[:, 5].astype(float)
        h = df.iloc[:, 6].astype(float)
        recs.append(pd.DataFrame({
            "slide_id": slide_ids,
            "tile_id": tile_ids,
            "x": x, "y": y, "w": w, "h": h,
            "cluster_name": [f"{name_prefix}:{hc}" for hc in hc_raw],
            "parent": name_prefix
        }))
    if recs:
        return pd.concat(recs, ignore_index=True)
    return pd.DataFrame(columns=["slide_id","tile_id","x","y","w","h","cluster_name","parent"])

def read_all_train_clusters(paths_cfg: dict) -> pd.DataFrame:
    recs = []
    cluster_dirs = paths_cfg.get("train_clusters", {}) or {}
    for prefix, dr in cluster_dirs.items():
        if os.path.isdir(dr):
            recs.append(read_cluster_dir_fixed(dr, prefix))
        else:
            logging.warning(f"[train_clusters] Directory not found: {dr} (skip)")
    trans_dirs = paths_cfg.get("transition_clusters", {}) or {}
    for prefix, dr in trans_dirs.items():
        if os.path.isdir(dr):
            recs.append(read_cluster_dir_fixed(dr, prefix))
        else:
            logging.warning(f"[transition_clusters] Directory not found: {dr} (skip)")
    if not recs:
        raise ValueError("No cluster CSVs were loaded from train/transition cluster directories.")
    return pd.concat(recs, ignore_index=True)

# ----------------------------- Build Stats ----------------------------- #

def build_consensus_stats(cfg: dict, embed_col: str, coord_cols: List[str]):
    logging.info("[A] Loading training embeddings …")
    df_embed = read_train_embeddings(cfg["paths"], embed_col)

    logging.info("[A] Loading training cluster labels …")
    df_clu = read_all_train_clusters(cfg["paths"])

    logging.info("[A] Merging embeddings with cluster labels …")
    df_c = pd.merge(
        df_embed[["slide_id", "tile_id", "embed", "parent"]],
        df_clu[["slide_id", "tile_id", "cluster_name", "parent"]],
        on=["slide_id", "tile_id", "parent"],
        how="inner"
    )
    if df_c.empty:
        raise ValueError("Merge of training embeddings and cluster labels is empty. Check join keys.")
    logging.info(f"[A] Training patches for modeling: {len(df_c):,}")

    stats = {}
    cluster_list = sorted(df_c["cluster_name"].unique().tolist())

    lam = float(cfg["algorithm"].get("shrink_lambda", 0.20))
    tau_mode = cfg["algorithm"].get("tau_mode", "iqr")
    tau_floor = float(cfg["algorithm"].get("tau_floor", 0.25))
    tau_const = float(cfg["algorithm"].get("tau_constant", 1.0))

    for cname in cluster_list:
        X = np.stack(df_c.loc[df_c["cluster_name"] == cname, "embed"].values)
        mu = X.mean(axis=0).astype(np.float32)
        S  = np.cov(X, rowvar=False)
        S += np.eye(S.shape[0], dtype=np.float32) * 1e-6
        S_sh = _shrink_cov(S, lam).astype(np.float32)
        invS = np.linalg.inv(S_sh).astype(np.float32)

        d = _mahalanobis(X, mu, invS)
        r95 = float(np.percentile(d, 95.0))
        if tau_mode == "iqr":
            tau = max(_iqr(d), tau_floor)
        else:
            tau = float(tau_const)

        stats[cname] = {"mu": mu, "invS": invS, "r95": r95, "tau": float(tau)}

    logging.info(f"[A] Total consensus clusters: {len(cluster_list)}")

    # KNN bank for EC/HGSC/Transitions
    mask_knn = df_c["parent"].isin(["Endo", "High", "Endo_to_High", "High_to_Endo"])
    df_knn = df_c.loc[mask_knn, ["embed", "cluster_name", "parent"]].copy()
    emb_bank = np.stack(df_knn["embed"].values) if len(df_knn) else np.zeros((0, len(df_c["embed"].iloc[0])), dtype=np.float32)
    clus_bank = df_knn["cluster_name"].astype(str).values if len(df_knn) else np.array([], dtype=str)
    parent_bank = df_knn["parent"].astype(str).values if len(df_knn) else np.array([], dtype=str)
    is_transition_bank = np.isin(parent_bank, ["Endo_to_High", "High_to_Endo"]) if len(df_knn) else np.array([], dtype=bool)

    trans_cluster_counts = pd.Series(clus_bank[is_transition_bank]).value_counts().to_dict() if len(clus_bank) else {}
    pi_trans = float(is_transition_bank.mean()) if len(is_transition_bank) else 0.0

    knn_bank = {
        "emb": emb_bank,
        "clus": clus_bank,
        "parent": parent_bank,
        "is_trans": is_transition_bank,
        "trans_counts": trans_cluster_counts,
        "pi_trans": pi_trans,
    }
    return stats, cluster_list, knn_bank

# ----------------------------- IO (Testing) ----------------------------- #

def read_test_embeddings(paths_cfg: dict, embed_col: str, coord_cols: List[str]) -> pd.DataFrame:
    """
    NEW: test CSVs must include column 'pred' (short code CC/EC/HGSC/SBT).
         Optional 'label' column is carried through if present.
    """
    dfs = []
    test_embeds = paths_cfg.get("test_embeds", {}) or {}
    if not test_embeds:
        raise ValueError("paths.test_embeds is empty in config.")

    for st, fp in test_embeds.items():
        if not os.path.exists(fp):
            logging.warning(f"[test] Embedding file not found: {fp} (skip)")
            continue
        df = pd.read_csv(fp)
        need = {"slide_id", "tile_id", embed_col, "pred"}
        if not need.issubset(df.columns):
            missing = sorted(need - set(df.columns))
            raise ValueError(f"[test] {fp} missing columns: {missing}")
        emb = np.stack(df[embed_col].apply(parse_embed_vec).values)
        emb = l2_normalize_rows(emb)

        keep = ["slide_id", "tile_id", "pred", "label"] + [c for c in coord_cols if c in df.columns]
        keep = [c for c in keep if c in df.columns]
        out = df[keep].copy()
        out["embed"] = list(emb)
        dfs.append(out)
        logging.info(f"[test] Loaded embeddings: {st} ({len(out):,} patches)")

    if not dfs:
        raise ValueError("No test embeddings were loaded.")
    return pd.concat(dfs, ignore_index=True)

# ----------------------------- each_slide_result ----------------------------- #

def _strict_subtype_series(s: pd.Series, mapping: Dict[str, str], colname: str) -> pd.Series:
    s_clean = s.astype(str).str.strip()
    unmapped = sorted(set(s_clean.unique()) - set(mapping.keys()))
    if unmapped:
        raise ValueError(f"[each_slide] Column '{colname}' contains unknown labels: {unmapped}\n"
                         f"Allowed values: {list(mapping.keys())}")
    return s_clean.map(mapping)

def load_true_pred_table(each_slide_csv: str, mapping: Dict[str, str]):
    df = pd.read_csv(each_slide_csv)
    df["true_short"] = _strict_subtype_series(df["name_label"], mapping, "name_label")
    df["pred_short"] = _strict_subtype_series(df["name_pred"],  mapping, "name_pred")
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
    algo_cfg: dict
):
    """
    df_slide is a SINGLE-PRED subset of one slide (we run this per pred).
    slide_group: "EC_HGSC"/"HGSC_EC" to enable hybrid; otherwise a true short label ("EC","HGSC","CC","SBT").
    """
    X = np.stack(df_slide["embed"].values)
    N, C = X.shape[0], len(cluster_list)

    Dmat = np.zeros((N, C), dtype=np.float32)
    tau_vec = np.zeros((C,), dtype=np.float32)
    r95_vec = np.zeros((C,), dtype=np.float32)

    for j, cname in enumerate(cluster_list):
        st = stats[cname]
        Dmat[:, j] = _mahalanobis(X, st["mu"], st["invS"])
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

    # Params
    knn_k = int(algo_cfg.get("knn_k", 30))
    knn_distance = str(algo_cfg.get("knn_distance", "cosine"))
    ptrans_abs_min = float(algo_cfg.get("ptrans_abs_min", 0.04))
    conf_min = float(algo_cfg.get("conf_min", 0.55))
    imbalance_alpha = float(algo_cfg.get("imbalance_alpha", 0.5))
    relax_rel = float(algo_cfg.get("outlier_relax_rel", 0.10))
    relax_abs = float(algo_cfg.get("outlier_relax_abs", 4.0))

    if use_mixture:
        relax_thr = np.maximum(nearest_r95 * (1.0 + relax_rel), nearest_r95 + relax_abs)
        outlier_relaxed = (d_min > relax_thr)
        outlier = outlier_relaxed
        assigned = np.where(outlier, "Outlier", nearest_cluster)

        is_core_EC = np.array([str(c).startswith("Endo:") for c in nearest_cluster], dtype=bool)
        is_core_HG = np.array([str(c).startswith("High:") for c in nearest_cluster], dtype=bool)
        candidate = (~outlier) & (is_core_EC | is_core_HG)

        pi_trans = float(knn_bank["pi_trans"])
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

            # EC↔HGSC uses absolute threshold only
            is_transition_like = (p_trans >= ptrans_abs_min)
            if not is_transition_like:
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

    return df_out  # NOTE: slide vector will be computed after we merge all preds

# ----------------------------- Main ----------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Morphologic Spectrum Soft Assignment (publish-ready, per-pred merge)")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    EMBED_COL = "embed"
    COORD_COLS = ["x", "y", "w", "h"]

    # A) training stats
    stats, cluster_list, knn_bank = build_consensus_stats(cfg, EMBED_COL, COORD_COLS)

    # B) test embeddings (MUST include 'pred')
    logging.info("[B] Loading test embeddings …")
    df_test = read_test_embeddings(cfg["paths"], EMBED_COL, COORD_COLS)

    # C) true/pred/correct lookup
    slide2true, slide2pred1, slide2corr = load_true_pred_table(
        cfg["paths"]["each_slide_result"],
        cfg["subtypes"]["long_to_short"]
    )

    # D) outputs
    out_root = Path(cfg["paths"]["output_dir"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    patch_csv_pattern = cfg["output"].get("patch_csv_pattern", "{slide_id}_patch_assign.csv")
    matrix_name = cfg["output"].get("matrix_name", "slide_cluster_matrix.csv")

    algo_cfg = cfg["algorithm"]
    min_patches_per_slide = int(algo_cfg.get("min_patches_per_slide", 10))

    # E) per slide: split by pred, assign, then merge to ONE CSV
    matrix_rows = []
    n_written = 0

    for sid, sub_all in df_test.groupby("slide_id"):
        if len(sub_all) < min_patches_per_slide:
            logging.warning(f"[warn] {sid}: total patches={len(sub_all)} < min_patches_per_slide")

        true_short: Optional[str] = slide2true.get(str(sid))
        if true_short is None:
            # fallback: if test CSV has a 'label' column that already is a short code
            if "label" in sub_all.columns and len(sub_all["label"].dropna()) > 0:
                true_short = str(sub_all["label"].iloc[0]).strip()
                logging.warning(f"[fallback] {sid}: using label='{true_short}' from test CSV")
            else:
                logging.warning(f"[skip] {sid}: not found in each_slide_result and no 'label' column – skip")
                continue

        per_pred_out = []
        for pred_name, sub_pred in sub_all.groupby("pred"):
            sub_pred = sub_pred.reset_index(drop=True)
            if len(sub_pred) == 0:
                continue

            # EC↔HGSC pair enables hybrid rule
            if {true_short, str(pred_name)} == {"Endo", "High",}:
                slide_group = "Endo_High" if true_short == "Endo" else "High_Endo"
            else:
                slide_group = true_short

            df_out_one_pred = soft_assign_one_slide(
                df_slide=sub_pred,
                stats=stats,
                cluster_list=cluster_list,
                knn_bank=knn_bank,
                slide_group=slide_group,
                algo_cfg=algo_cfg
            )
            # ensure 'pred' present (safety)
            if "pred" not in df_out_one_pred.columns:
                df_out_one_pred["pred"] = pred_name
            per_pred_out.append(df_out_one_pred)

        if not per_pred_out:
            logging.warning(f"[skip] {sid}: no valid pred subset outputs")
            continue

        df_merged = pd.concat(per_pred_out, ignore_index=True)

        # Write ONE CSV under true_short folder
        group_dir = out_root / true_short
        group_dir.mkdir(parents=True, exist_ok=True)
        patch_fp = group_dir / patch_csv_pattern.format(slide_id=sid)
        df_merged.to_csv(patch_fp, index=False)
        logging.info(f"[OK] Wrote patch-level assignment (merged preds): {patch_fp} (rows={len(df_merged)})")
        n_written += 1

        # Build slide×cluster vector aggregated over ALL assigned (incl. Outlier), normalized
        counts = df_merged["assigned_cluster"].astype(str).value_counts(normalize=True)
        row = {"slide_id": sid, "group": true_short}
        for c in cluster_list:
            row[c] = float(counts.get(c, 0.0))
        row["Outlier"] = float(counts.get("Outlier", 0.0))
        matrix_rows.append(row)

    if n_written == 0:
        logging.warning("[WARN] No slides were written – please check test_embeds / each_slide_result / inputs")
        return

    # F) write matrix (wide)
    if matrix_rows:
        col_order = ["slide_id", "group"] + cluster_list + ["Outlier"]
        mat_wide_df = pd.DataFrame(matrix_rows, columns=col_order)
        mat_fp = out_root / matrix_name
        mat_wide_df.to_csv(mat_fp, index=False)
        logging.info(f"[OK] Wrote slide×cluster matrix: {mat_fp}")

if __name__ == "__main__":
    main()
