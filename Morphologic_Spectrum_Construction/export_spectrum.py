from __future__ import annotations

import os
import glob
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ----------------------------- Small utilities ----------------------------- #

def parse_embed_vec(s: str) -> np.ndarray:
    """Parse a semicolon-separated embedding string into a float32 vector."""
    return np.array([float(x) for x in str(s).split(";") if x != ""], dtype=np.float32)


def l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2D array."""
    norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norm


def _shrink_cov(S: np.ndarray, lam: float) -> np.ndarray:
    """Shrinkage of covariance matrix toward its diagonal."""
    diag = np.diag(np.diag(S))
    return (1.0 - lam) * S + lam * diag


def _mahalanobis(X: np.ndarray, mu: np.ndarray, invS: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of rows of X to mean mu with covariance invS."""
    diff = X - mu[None, :]
    return np.sqrt(np.sum(diff @ invS * diff, axis=1))


def _iqr(arr: np.ndarray) -> float:
    """Inter-quartile range of a 1D array."""
    q75, q25 = np.percentile(arr, [75, 25])
    return float(max(q75 - q25, 1e-6))


# ----------------------------- IO: core / transition embeddings ----------------------------- #

def read_core_and_transition_embeddings(
    core_embeds: Dict[str, str | Path],
    transition_embeds: Optional[Dict[str, str | Path]],
    embed_col: str,
) -> pd.DataFrame:
    """
    Read high-contribution patch embeddings for core subtypes and (optionally)
    transition phenotypes.

    Args
    ----
    core_embeds:
        Mapping from core subtype name (e.g. "Clear", "Endo") to CSV path.
    transition_embeds:
        Mapping from transition phenotype name (e.g. "Endo_to_High") to CSV path,
        or None if no transition phenotypes are used.
    embed_col:
        Name of the column containing semicolon-separated embeddings.

    Returns
    -------
    df : DataFrame with columns:
        - slide_id
        - tile_id
        - embed  (np.ndarray)
        - parent (core subtype or transition name)
    """
    dfs: List[pd.DataFrame] = []

    # Core Patterns
    if not core_embeds:
        raise ValueError("core_embeds is empty; at least one core embedding CSV is required.")

    for parent, fp in core_embeds.items():
        fp_str = str(fp)
        if not os.path.exists(fp_str):
            logging.warning(f"[core_embeds] Embedding file not found: {fp_str} (skip)")
            continue

        df = pd.read_csv(fp_str)
        need = {"slide_id", "tile_id", embed_col}
        if not need.issubset(df.columns):
            missing = need - set(df.columns)
            raise ValueError(f"[core_embeds] {fp_str} missing columns: {missing}")

        emb = np.stack(df[embed_col].apply(parse_embed_vec).values)
        emb = l2_normalize_rows(emb)

        out = df[["slide_id", "tile_id"]].copy()
        out["embed"] = list(emb)
        out["parent"] = parent  # Clear / Endo / High / Border
        dfs.append(out)
        logging.info(f"[core_embeds] Loaded core embeddings: {parent} ({len(out):,} patches)")

    # Transition Patterns (optional)
    if transition_embeds:
        for parent, fp in transition_embeds.items():
            fp_str = str(fp)
            if not os.path.exists(fp_str):
                logging.warning(f"[transition_embeds] Embedding file not found: {fp_str} (skip)")
                continue

            df = pd.read_csv(fp_str)
            need = {"slide_id", "tile_id", embed_col}
            if not need.issubset(df.columns):
                missing = need - set(df.columns)
                raise ValueError(f"[transition_embeds] {fp_str} missing columns: {missing}")

            emb = np.stack(df[embed_col].apply(parse_embed_vec).values)
            emb = l2_normalize_rows(emb)

            out = df[["slide_id", "tile_id"]].copy()
            out["embed"] = list(emb)
            out["parent"] = parent  # Endo_to_High / High_to_Endo
            dfs.append(out)
            logging.info(f"[transition_embeds] Loaded transition embeddings: {parent} ({len(out):,} patches)")

    if not dfs:
        raise ValueError("No training embeddings were loaded from core_embeds / transition_embeds.")

    return pd.concat(dfs, ignore_index=True)


# ----------------------------- IO: core / transition clusters ----------------------------- #

def read_cluster_dir_fixed(dirpath: str | Path, name_prefix: str) -> pd.DataFrame:
    """
    Read all cluster CSVs under a directory and return a single DataFrame.
    """
    recs: List[pd.DataFrame] = []
    dirpath_str = str(dirpath)

    for fp in glob.glob(os.path.join(dirpath_str, "*.csv")):
        df = pd.read_csv(fp, header=None)
        if df.shape[1] < 7:
            raise ValueError(f"[core_clusters] {fp} has <7 columns")
        slide_ids = df.iloc[:, 0].astype(str).str.replace(".svs", "", regex=False)
        tile_ids = df.iloc[:, 1].astype(str)
        hc_raw = df.iloc[:, 2].astype(str).str.strip().str.lower()
        x = df.iloc[:, 3].astype(float)
        y = df.iloc[:, 4].astype(float)
        w = df.iloc[:, 5].astype(float)
        h = df.iloc[:, 6].astype(float)
        recs.append(pd.DataFrame({
            "slide_id": slide_ids,
            "tile_id": tile_ids,
            "x": x, "y": y, "w": w, "h": h,
            "cluster_name": [f"{name_prefix}:{hc}" for hc in hc_raw],
            "parent": name_prefix,
        }))

    if recs:
        return pd.concat(recs, ignore_index=True)

    return pd.DataFrame(columns=["slide_id", "tile_id", "x", "y", "w", "h", "cluster_name", "parent"])


def read_all_core_and_transition_clusters(
    core_clusters: Dict[str, str | Path],
    transition_clusters: Optional[Dict[str, str | Path]],
) -> pd.DataFrame:
    """
    Read cluster CSVs for core subtypes and (optionally) transition phenotypes.

    Args
    ----
    core_clusters:
        Mapping from core subtype name (e.g. "Clear", "Endo") to cluster directory.
    transition_clusters:
        Mapping from transition phenotype name (e.g. "Endo_to_High") to cluster
        directory, or None if no transition clusters are used.

    Returns
    -------
    df : DataFrame with columns:
        - slide_id
        - tile_id
        - x, y, w, h
        - cluster_name (e.g. 'Endo:hc_0', 'High_to_Endo:hc_3')
        - parent       (core subtype or transition name)
    """
    recs: List[pd.DataFrame] = []

    # Core clusters
    if not core_clusters:
        raise ValueError("core_clusters is empty; at least one cluster directory is required.")

    for prefix, dr in core_clusters.items():
        dr_str = str(dr)
        if os.path.isdir(dr_str):
            recs.append(read_cluster_dir_fixed(dr_str, prefix))
        else:
            logging.warning(f"[core_clusters] Directory not found: {dr_str} (skip)")

    # Transition clusters (optional)
    if transition_clusters:
        for prefix, dr in transition_clusters.items():
            dr_str = str(dr)
            if os.path.isdir(dr_str):
                recs.append(read_cluster_dir_fixed(dr_str, prefix))
            else:
                logging.warning(f"[transition_clusters] Directory not found: {dr_str} (skip)")

    if not recs:
        raise ValueError("No cluster CSVs were loaded from core_clusters / transition_clusters.")

    return pd.concat(recs, ignore_index=True)


# ----------------------------- Build consensus stats ----------------------------- #

def build_consensus_stats(
    core_embeds: Dict[str, str | Path],
    core_clusters: Dict[str, str | Path],
    transition_embeds: Optional[Dict[str, str | Path]],
    transition_clusters: Optional[Dict[str, str | Path]],
    algorithm_cfg: dict,
    embed_col: str = "embed",
):
    """
    Build consensus morphologic spectrum statistics from core & transition embeddings
    and cluster assignments.

    Args
    ----
    core_embeds:
        Mapping from core subtype name (Clear/Endo/High/Border) to embedding CSV.
    core_clusters:
        Mapping from core subtype name to cluster directory ('slide_results').
    transition_embeds:
        Mapping from transition phenotype name (Endo_to_High/High_to_Endo) to
        embedding CSV, or None if not used.
    transition_clusters:
        Mapping from transition phenotype name to cluster directory, or None.
    algorithm_cfg:
        Dictionary of algorithm hyperparameters, expected keys include:
        - shrink_lambda
        - tau_mode
        - tau_floor
        - tau_constant
    embed_col:
        Name of the embedding column in the CSV files.

    Returns
    -------
    stats: dict
        cluster_name -> {"mu", "invS", "r95", "tau"}
    cluster_list: list[str]
        Sorted list of all cluster names included in the spectrum.
    knn_bank: dict
        Bank used for EC↔HGSC transition detection in Stage 2 mapping.
    """
    logging.info("[A] Loading core/transition embeddings …")
    df_embed = read_core_and_transition_embeddings(core_embeds, transition_embeds, embed_col)

    logging.info("[A] Loading core/transition cluster labels …")
    df_clu = read_all_core_and_transition_clusters(core_clusters, transition_clusters)

    logging.info("[A] Merging embeddings with cluster labels …")
    df_c = pd.merge(
        df_embed[["slide_id", "tile_id", "embed", "parent"]],
        df_clu[["slide_id", "tile_id", "cluster_name", "parent"]],
        on=["slide_id", "tile_id", "parent"],
        how="inner",
    )
    if df_c.empty:
        raise ValueError("Merge of embeddings and cluster labels is empty. Check join keys and inputs.")
    logging.info(f"[A] Training patches for modeling: {len(df_c):,}")

    stats: Dict[str, dict] = {}
    cluster_list: List[str] = sorted(df_c["cluster_name"].unique().tolist())

    lam = float(algorithm_cfg.get("shrink_lambda", 0.20))
    tau_mode = algorithm_cfg.get("tau_mode", "iqr")
    tau_floor = float(algorithm_cfg.get("tau_floor", 0.25))
    tau_const = float(algorithm_cfg.get("tau_constant", 1.0))

    for cname in cluster_list:
        X = np.stack(df_c.loc[df_c["cluster_name"] == cname, "embed"].values)
        mu = X.mean(axis=0).astype(np.float32)
        S = np.cov(X, rowvar=False)
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

    if len(df_knn):
        emb_bank = np.stack(df_knn["embed"].values)
        clus_bank = df_knn["cluster_name"].astype(str).values
        parent_bank = df_knn["parent"].astype(str).values
        is_transition_bank = np.isin(parent_bank, ["Endo_to_High", "High_to_Endo"])

        trans_cluster_counts = pd.Series(clus_bank[is_transition_bank]).value_counts().to_dict()
        pi_trans = float(is_transition_bank.mean())
    else:
        emb_bank = np.zeros((0, len(df_c["embed"].iloc[0])), dtype=np.float32)
        clus_bank = np.array([], dtype=str)
        parent_bank = np.array([], dtype=str)
        is_transition_bank = np.array([], dtype=bool)
        trans_cluster_counts = {}
        pi_trans = 0.0

    knn_bank = {
        "emb": emb_bank,
        "clus": clus_bank,
        "parent": parent_bank,
        "is_trans": is_transition_bank,
        "trans_counts": trans_cluster_counts,
        "pi_trans": pi_trans,
    }

    return stats, cluster_list, knn_bank


# ----------------------------- Export wrapper ----------------------------- #

def export_spectrum_stats(
    core_embeds: Dict[str, str | Path],
    core_clusters: Dict[str, str | Path],
    transition_embeds: Optional[Dict[str, str | Path]],
    transition_clusters: Optional[Dict[str, str | Path]],
    algorithm_cfg: dict,
    output_path: Path,
    embed_col: str = "embed",
) -> Path:
    """
    High-level wrapper for Stage 1 – Step 8:
    build morphologic spectrum statistics and export them as a single file.

    Args
    ----
    core_embeds:
        Core subtype embedding CSVs (Step 6 outputs):
        { "Clear": Clear.csv, "Endo": Endo.csv, "High": High.csv, "Border": Border.csv }.
    core_clusters:
        Core subtype cluster directories (Step 7 outputs: slide_results folders).
    transition_embeds:
        Transition phenotype embedding CSVs (optional).
    transition_clusters:
        Transition phenotype cluster directories (optional).
    algorithm_cfg:
        Algorithm hyperparameters (e.g., cfg["algorithm"] from config.yaml).
    output_path:
        Where to save the exported stats file (.pkl is recommended).
    embed_col:
        Column name for embeddings in the CSVs.

    Returns
    -------
    output_path : Path
        Path to the exported spectrum stats file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("[export] Building consensus morphologic spectrum stats …")
    stats, cluster_list, knn_bank = build_consensus_stats(
        core_embeds=core_embeds,
        core_clusters=core_clusters,
        transition_embeds=transition_embeds,
        transition_clusters=transition_clusters,
        algorithm_cfg=algorithm_cfg,
        embed_col=embed_col,
    )

    payload = {
        "stats": stats,
        "cluster_list": cluster_list,
        "knn_bank": knn_bank,
    }

    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    logging.info(f"[export] Exported morphologic spectrum stats to: {output_path}")
    return output_path
