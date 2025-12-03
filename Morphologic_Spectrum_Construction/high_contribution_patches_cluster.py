import yaml
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
import seaborn as sns, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
import faiss
import sys
from pathlib import Path
import argparse

try:
    from .high_contribution_patches_extraction import split_feature_tile_id
    from .train_early_stopping import get_feature_bag_path
except ImportError:
    from high_contribution_patches_extraction import split_feature_tile_id
    from train_early_stopping import get_feature_bag_path

# -------------------- Utility functions --------------------
def parse_embed_array(embed_str):
    return np.array([float(x) for x in embed_str.split(';') if x], dtype=np.float32)

def extract_slide_id(s: str) -> str:
    return Path(s).stem  # 'ANONXYZ1_1_1'

def export_hc_cluster_slide_metadata(df_sub, mapping_csv_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(mapping_csv_path, dtype=str)

    mapping['slide_id'] = mapping['SVS Filename'].apply(extract_slide_id)

    slide2person = dict(zip(mapping['slide_id'], mapping['Person ID']))

    df = df_sub.copy()

    df['person_id'] = df['slide_id'].map(slide2person)

    miss = df[df['person_id'].isna()]['slide_id'].unique()
    if len(miss):
        print(f"Warning: {len(miss)} slide_id not found in metadata mapping: {miss[:5]} ...")

    summary = (
        df
        .groupby(['hc_label','slide_id','label','person_id'])
        .agg(patch_count=('tile_id','nunique'))
        .reset_index()
    )

    out_csv = out_dir/"cluster_slide_summary.csv"
    summary.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"Exported summary table: {out_csv}")

def export_hc_cluster_coords_and_paths(df_sub, slide_h5_coords_basepath, out_dir):
    output_dir = out_dir/'slide_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for slide_id, grp in df_sub.groupby("slide_id"):
        print(f"Exporting patch corrdinates of {slide_id}")
        # a) Load original patch coords
        h5_path = get_feature_bag_path(slide_h5_coords_basepath, slide_id)
        _,coords, orig_ids = split_feature_tile_id(h5_path)
        
        # id -> coord mapping
        id2coord = {str(tid).strip(): tuple(c) for c, tid in zip(coords, orig_ids)}
        for tid in sorted(id2coord):
            print(tid, id2coord[tid])
        
        # 2) Collect all hc_label records for this slide
        records = []
        for _, row in grp.iterrows():
            hc_label = row["hc_label"]
            tid      = str(row["tile_id"]).strip()
            minx, miny, maxx, maxy = id2coord[tid]
            records.append({
                "slide_id": f"{slide_id}.svs",
                "tile_id":  tid,
                "hc_label": f"hc_{hc_label}",
                "x":        minx,
                "y":        miny,
                "width":    maxx - minx,
                "height":   maxy - miny
            })
        # Write file
        if records:
                df_out = pd.DataFrame(
                    records,
                    columns=["slide_id","tile_id","hc_label","x","y","width","height"]
                )
                csv_path = output_dir / f"{slide_id}_hc_coords.csv"
                df_out.to_csv(csv_path, index=False, header=False)
                print(f"→ Saved {slide_id} HC patches to {csv_path}")
        else:
            print(f"[WARN] Slide {slide_id} has no HC patches, skipping.")

def load_and_preprocess(csv_path, disease):
    df = pd.read_csv(csv_path)
    df['embed_array'] = df['embed'].apply(parse_embed_array)
    df_sub = df[df['label']==disease].copy()
    df_sub['embed_tuple'] = df_sub['embed_array'].apply(lambda a: tuple(a.tolist()))
    df_sub = df_sub.drop_duplicates(['slide_id', 'tile_id', 'embed_tuple'])
    features = np.vstack(df_sub['embed_array'].to_numpy())
    weights = df_sub['weight'].to_numpy()
    return df_sub, features, weights


def compute_micro_clusters(features, n_micro=2000, niter=20):
    d = features.shape[1]
    kmeans = faiss.Kmeans(d, n_micro, niter=niter, verbose=False)
    kmeans.train(features.astype(np.float32))
    return kmeans.centroids


def build_consensus_matrix(X, k, n_runs=30, p_item=0.8):
    n = X.shape[0]
    C = np.zeros((n, n), dtype=np.float32)
    rng = np.random.RandomState(123)
    for r in range(n_runs):
        idx = rng.choice(n, int(n*p_item), replace=False)
        sub = X[idx]
        labels = KMeans(n_clusters=k, random_state=r).fit_predict(sub)
        for i, ii in enumerate(idx):
            same = idx[labels[i]==labels]
            C[ii, same] += 1
    C /= n_runs
    np.fill_diagonal(C, 1.0)
    return C


def cophenetic_corr_from_consensus(C):
    D = 1 - C
    dist_vec = pdist(D)
    Zc = linkage(dist_vec, method='average')
    coph, _ = cophenet(Zc, dist_vec)
    return coph


def select_best_k_consensus(features, k_list, reps, p_item, n_micro):
    micro = compute_micro_clusters(features, n_micro=n_micro)
    scores = {}
    for k in k_list:
        C = build_consensus_matrix(micro, k, n_runs=reps, p_item=p_item)
        rho = cophenetic_corr_from_consensus(C)
        print(f"k={k} -> cophenetic_corr={rho:.3f}")
        scores[k] = rho
    best_k = max(scores, key=scores.get)
    print(f"==> Best k by consensus = {best_k}")
    return best_k, scores


def cluster_high_contribution_patches(
    metadata_csv,
    high_contri_csv,
    disease,
    mode,
    out_dir,
    feature_bag_dir=None,
    cluster_list=None,
    reps: int = 30,
    p_item: float = 0.8,
    n_micro: int = 1000,
    n_clusters: int | None = None,
):
    """
    High-level API to cluster high-contribution patches and optionally export
    QuPath-ready patch coordinates.

    Parameters
    ----------
    metadata_csv : str or Path
        Path to the cohort metadata CSV
        (must include 'Person ID', 'Tissue ID', 'SVS Filename').
    high_contri_csv : str or Path
        CSV of high-contribution patches (output from high_contribution_patches_extraction),
        must include columns: slide_id, label, tile_id, weight, embed.
    disease : str
        Target disease/subtype label to cluster (e.g., "Clear", "Endo", ...).
    mode : {"evaluate_clusters", "export"}
        - "evaluate_clusters": evaluate multiple candidate k using consensus clustering.
        - "export": use a fixed --n_clusters to cluster all patches and export coordinates.
    out_dir : str or Path
        Output directory; a subfolder named after `disease` will be created inside it.
    feature_bag_dir : str or Path, optional (required in "export" mode)
        Directory containing all *_features.h5 files (feature bags).
    cluster_list : list[int], optional
        Candidate k values for consensus evaluation (used only in "evaluate_clusters" mode).
    reps : int, default=30
        Number of subsampling runs for consensus clustering.
    p_item : float, default=0.8
        Subsampling fraction per run (0–1].
    n_micro : int, default=1000
        Number of FAISS micro-centroids to compute before consensus.
    n_clusters : int or None, optional
        Number of clusters in "export" mode (required there).

    Returns
    -------
    None
        Results are written to disk:
        - evaluate_clusters:
            * consensus_k_selection.csv
            * consensus_heatmap_k{k}.png
        - export:
            * cluster_slide_summary.csv
            * slide_results/<slide_id>_hc_coords.csv
    """
    metadata_csv = Path(metadata_csv)
    high_contri_csv = Path(high_contri_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    disease = str(disease)
    mode = str(mode)

    # Disease-specific subfolder
    output_dir = out_dir / disease
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess patch embeddings
    df_sub, features, weights = load_and_preprocess(high_contri_csv, disease)

    if mode == "evaluate_clusters":
        if not cluster_list:
            raise ValueError("--cluster_list must contain at least one candidate k value.")
        if not (0 < p_item <= 1):
            raise ValueError("--p_item must be a float in the range (0, 1].")
        if n_micro <= 0:
            raise ValueError("--n_micro must be a positive integer.")
        if reps <= 0:
            raise ValueError("--reps must be a positive integer.")

        best_k, scores = select_best_k_consensus(
            features, cluster_list, reps, p_item, n_micro
        )
        # Save scores table
        df_res = pd.DataFrame(
            [{"k": k, "cophenetic_corr": v} for k, v in scores.items()]
        )
        df_res.to_csv(output_dir / "consensus_k_selection.csv", index=False)
        print(f"[INFO] Saved consensus k-selection table to {output_dir/'consensus_k_selection.csv'}")

        # Plot heatmaps for each k
        micro = compute_micro_clusters(features, n_micro=n_micro)
        for k in cluster_list:
            print(f"[INFO] Plotting consensus heatmap for k={k}")
            C = build_consensus_matrix(micro, k, n_runs=reps, p_item=p_item)
            Zc = linkage(1 - C, method="average", metric="euclidean")
            g = sns.clustermap(
                C,
                row_linkage=Zc,
                col_linkage=Zc,
                cmap="Blues",
                vmin=0,
                vmax=1,
                figsize=(6, 6),
            )
            out_png = output_dir / f"consensus_heatmap_k{k}.png"
            g.savefig(out_png, dpi=150)
            plt.close(g.figure)
            print(f"[INFO] Saved consensus heatmap for k={k} to {out_png}")
        return

    elif mode == "export":
        if n_clusters is None:
            raise ValueError("--n_clusters is required in 'export' mode.")
        if feature_bag_dir is None:
            raise ValueError("--feature_bag_dir is required in 'export' mode.")
        feature_bag_dir = Path(feature_bag_dir)

        # Perform complete linkage clustering on full feature set
        print(f"[INFO] Performing complete-linkage clustering with k={n_clusters}")
        Z = linkage(features, method="complete", metric="euclidean")
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        df_sub["hc_label"] = labels

        # Export cluster summaries
        export_hc_cluster_slide_metadata(df_sub, metadata_csv, output_dir)

        # Export patch-level coordinates for each slide (for QuPath import)
        export_hc_cluster_coords_and_paths(df_sub, feature_bag_dir, output_dir)

        print(f"[INFO] Exported clusters for k={n_clusters} using complete linkage.")
        return

    else:
        raise ValueError(f"Unknown mode: {mode}")
    


def main(args):
    cluster_high_contribution_patches(
        metadata_csv=args.metadata_csv,
        high_contri_csv=args.high_contri_csv,
        disease=args.disease,
        mode=args.mode,
        out_dir=args.out_dir,
        feature_bag_dir=args.feature_bag_dir,
        cluster_list=args.cluster_list,
        reps=args.reps,
        p_item=args.p_item,
        n_micro=args.n_micro,
        n_clusters=args.n_clusters,
    )
    # h5_base_path = args.feature_bag_dir
    # mapping_csv = args.metadata_csv

    # output_dir = args.out_dir/args.disease
    # output_dir.mkdir(parents=True, exist_ok=True)

    # df_sub, features, weights = load_and_preprocess(
    #     args.high_contri_csv, args.disease)

    # if args.mode == 'evaluate_clusters':
    #     best_k, scores = select_best_k_consensus(
    #         features, args.cluster_list, args.reps, args.p_item, args.n_micro)
    #     df_res = pd.DataFrame([{'k':k,'cophenetic_corr':v} for k,v in scores.items()])
    #     df_res.to_csv(output_dir/'consensus_k_selection.csv', index=False)

    #     for k in args.cluster_list:
    #         print(f"Plotting consensus heatmap for k={k}")
    #         micro = compute_micro_clusters(features, n_micro=args.n_micro)
    #         C = build_consensus_matrix(micro, k, n_runs=args.reps, p_item=args.p_item)
    #         Zc = linkage(1 - C, method='average', metric='euclidean')
    #         g = sns.clustermap(
    #             C,
    #             row_linkage=Zc,
    #             col_linkage=Zc,
    #             cmap='Blues',
    #             vmin=0, vmax=1,
    #             figsize=(6, 6)
    #         )
    #         out_png = output_dir/f'consensus_heatmap_k{k}.png'
    #         g.savefig(out_png, dpi=150)
    #         plt.close(g.figure)
    #         print(f"Saved consensus heatmap for k={k} to {out_png}")
    #     return

    # if args.mode == "export":
    #     best_k = args.n_clusters
    #     # Perform complete linkage clustering
    #     Z = linkage(features, method='complete', metric='euclidean')
    #     labels = fcluster(Z, t=best_k, criterion='maxclust')
    #     df_sub['hc_label'] = labels
        
    #     # Export cluster summaries
    #     export_hc_cluster_slide_metadata(df_sub,mapping_csv,output_dir)

    #     # Export patch-level coordinates for each slide (for QuPath import)
    #     export_hc_cluster_coords_and_paths(df_sub, h5_base_path, output_dir)

    #     print(f"Exported clusters for k={best_k} using complete linkage.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_csv",
        type=str,
        help="Path to the cohort metadata CSV (should include 'Person ID', 'Tissue ID', and 'SVS Filename' columns).",
        required=True,
    )
    parser.add_argument(
        "--feature_bag_dir",
        type=str,
        help="Directory containing all *_features.h5 files (feature bags).",
    )
    parser.add_argument(
        '--high_contri_csv', 
        type=Path, 
        help="Input CSV of high-contribution patches (must include columns: slide_id, label, tile_id, weight, embed).",
        required=True
    )
    parser.add_argument(
        '--disease', 
        type=str, 
        help="Target disease/subtype to cluster (e.g., Endo, High, ...).",
        required=True
    )
    parser.add_argument(
        '--mode', 
        choices=['evaluate_clusters','export'], 
        help="Mode: 'evaluate_clusters' to score k via consensus; 'export' to cluster and export coordinates/plots.",
        required=True
    )
    parser.add_argument(
        '--out_dir', 
        type=Path, 
        help="Output directory (will create subfolders per disease).",
        required=True
    )
    parser.add_argument(
        '--cluster_list', 
        nargs='+', 
        type=int, 
        help="Candidate k list for consensus evaluation (evaluate_clusters mode).",
        default=[2,3,4,5,6,7,8],
    )
    parser.add_argument(
        '--reps', 
        type=int, 
        default=30,
        help="Number of repetitions (subsampling runs) for consensus.",
    )
    parser.add_argument(
        '--p_item', 
        type=float, 
        default=0.8,
        help="Subsampling fraction per run for consensus (0–1]."
    )
    parser.add_argument(
        '--n_micro', 
        type=int, 
        default=1000,
        help="Number of FAISS micro-centroids to compute before consensus."
    )
    parser.add_argument(
        '--n_clusters', 
        type=int, 
        help="Number of clusters (k) to use when cutting the dendrogram. "
        "This parameter is only used in 'export' mode, where it is required. "
        "In 'evaluate_clusters' mode, the script will ignore this value and instead "
        "test multiple k values from --cluster_list to help select the best k."
    )
    args = parser.parse_args()

    def require(cond: bool, msg: str):
        if not cond:
            parser.error(msg)
    
    if args.mode == "evaluate_clusters":
        require(len(args.cluster_list) > 0,
                "--cluster_list must contain at least one candidate k value.")
        require(0 < args.p_item <= 1,
                "--p_item must be a float in the range (0, 1].")
        require(args.n_micro > 0,
                "--n_micro must be a positive integer (number of FAISS micro-centroids).")
        require(args.reps > 0,
                "--reps must be a positive integer (number of consensus runs).")

    elif args.mode == "export":
        require(args.n_clusters is not None,
                "--n_clusters is required in export mode (number of clusters to cut the dendrogram).")
        require(args.feature_bag_dir is not None,
                "--feature_bag_dir is required in export mode.")
        
        

    else:
        parser.error(f"Unknown mode: {args.mode}")
    main(args)
