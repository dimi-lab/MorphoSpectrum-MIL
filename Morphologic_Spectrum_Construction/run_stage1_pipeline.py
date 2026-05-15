import os

import pandas as pd
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yaml

from data_split_by_patient import run_patient_level_split

from preprocess import preprocess_wsi_to_h5

from train_early_stopping import train_single_round

from test import evaluate_all_rounds

from sample_stratification import run_sample_stratification

from high_contribution_patches_extraction import extract_high_contribution_patches

#from high_contribution_patches_cluster import cluster_high_contribution_patches

from high_contribution_patches_cluster import cluster_high_contribution_patches_ext


# ==================== 1. Data Preparation ====================
def data_preparation_func(metadata_csv, text=None):
    print("\n  This function creates the patient-level train/test splits used in MorphoXAI’s prediction-stability–based sample stratification procedure.")

    #get_input = input("\n  Input metadata csv filename: ").strip()
    #if get_input is None:
    #    raise ValueError("  Must input metadata file name.")
    #metafilename = get_input

    split_num = 5
    fold_num = 10
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    output_dir = "./Data_Split"
    label_config = "config.yaml"
    #meatdata_csv=f"{output_dir}/{metafilename}"

    print("\n  Now, start data split.... ")

    try:
        run_patient_level_split(data_csv=metadata_csv,
            output_dir=output_dir,
            label_config=label_config,
            mode="random",
            split_num=split_num,
            fold_num=fold_num,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
    except Exception as e:
        print(f"Function [run_patient_level_split] Failed: error={e}", flush=True)

    output_dir = Path(output_dir)
    
    split_files = list(output_dir.glob("Datasplit_*.csv"))
    if len(split_files) == 0:
        raise ValueError(f"No CSV files found in directory: {output_dir}")

    split_files = sorted(split_files)
    print("\n  Completed. Generated the following data split files:")
    for i in range(len(split_files)):
        print(f"    {split_files[i].stem}.csv")

    return metadata_csv


# ==================== 2. WSI feature_bags ====================
def feature_bags_func_all(csv_path):
    print("\n  This function uses CONCH to embed each tissue patch and saves the result to an `.h5` file.")
    print("  A corresponding QC image is also generated with the suffix `_features_QC.png`.")

    wsi_dir = "./wsi"
    output_dir = "./feature_bags"
    out_size = 224
    batch_size = 256
    tile_size_microns=360

    wsi_dir = Path(wsi_dir)
    output_dir = Path(output_dir)

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        slide_path = row["SVS Filename"]

        input_slide = Path(wsi_dir/slide_path)
        slide_id, _ = os.path.splitext(os.path.basename(str(input_slide)))

        output_file = Path(output_dir) / f"{slide_id}_features.h5"
        if output_file.exists():
            print(f"  Skip existing: {slide_id}")
            continue

        print(f"\n  Processing {slide_path} ")

        
        try:
            preprocess_wsi_to_h5(
                input_slide=input_slide,
                output_dir=output_dir,
                tile_size_microns=tile_size_microns,
                out_size=out_size,
                batch_size=batch_size,
                workers=1,
                hf_auth_token="",  # or rely on HF_TOKEN env var
            )
        except Exception as e:
            print(f"Function [preprocess_wsi_to_h5] Failed: error={e}", flush=True)

        print(f"  Completed. Generated {slide_id}_features.h5 ")


# ==================== 3. MIL training ====================
def MIL_training_func_all(text=None):
    print("\n  MorphoXAI adopts the CLAM framework as the MIL model.")
    print("  This function implements a k-fold cross-validation routine and applies early stopping. The best model checkpoint for every fold is saved in `./runs`.")

    fold_num = 10
    datasplit_dir = "./Data_Split"
    out_checkpoint_dir = Path("./runs")
    feature_bag_dir = "./feature_bags"
    label_config = "config.yaml"

    print("\n  Found data split files. All split files will be used for training.")
    glob_pattern="Datasplit_*_*_fold_by_patient.csv"
    datasplit_dir = Path(datasplit_dir)
    csv_files = [
        f.resolve()
        for f in datasplit_dir.glob(glob_pattern)
    ]
    if not csv_files:
        raise RuntimeError(
            f"[ERROR] No per-slide result CSVs found under {datasplit_dir} "
            f"matching pattern: {glob_pattern}"
        )

    for csv_path in csv_files:
        print(f"  Training use split file: {csv_path.stem}.csv")

        for fold_idx in range(fold_num):
            print(f"  Running round: {fold_idx}")
            
            try:
                train_single_round(
                    manifest=csv_path,
                    feature_bag_dir=feature_bag_dir,
                    out_checkpoint_dir=out_checkpoint_dir,
                    round_idx=fold_idx,
                    workers=1,
                    config_path=label_config,
                )
            except Exception as e:
                print(f"Function [train_single_round] Failed: error={e}", flush=True)

    print("\n  MIL training completed.")


# ==================== 4. MIL testing ====================
def MIL_testing_func_all(text=None):
    print("\n  This function evaluates trained MIL models on the held-out test sets produced during cross-validation.")

    fold_num = 10
    datasplit_dir = "./Data_Split"
    checkpoint_dir = "./runs"
    feature_bag_dir = "./feature_bags"
    label_config = "config.yaml"

    print("\n  Check and found data split files, and use all data split file for training.")
    glob_pattern="Datasplit_*_*_fold_by_patient.csv"
    datasplit_dir = Path(datasplit_dir)
    csv_files = [
        f.resolve()
        for f in datasplit_dir.glob(glob_pattern)
    ]
    if not csv_files:
        raise RuntimeError(
            f"[ERROR] No per-slide result CSVs found under {datasplit_dir} "
            f"matching pattern: {glob_pattern}"
        )

    for csv_path in csv_files:
        print(f"  Testing use split file: {csv_path.stem}.csv")

        
        try:
            evaluate_all_rounds(
                manifest=csv_path,
                feature_bag_dir=feature_bag_dir,
                checkpoint_dir=checkpoint_dir,
                round_num=fold_num,
                workers=1,
                config_path=label_config,
            )
        except Exception as e:
            print(f"Function [evaluate_all_rounds] Failed: error={e}", flush=True)

    print("\n  MIL testing completed.")


# ==================== 5. Sample Stratification ====================
def Sample_Stratification_func(metadata_csv, text=None):
    print("\n  This function aggregates **per-slide prediction results** across all shuffles and cross-validation folds, and then assigns each slide to one of three stability categories..")

    print(f"  Use metadata csv file: {metadata_csv}")

    output_dir = "./test_result"
    label_config = "config.yaml"

    print("\n  Starting processing.... ")

    try:
        run_sample_stratification(
            test_result_root=output_dir,
            metadata_csv=metadata_csv,
            out_dir=output_dir,
            config_path=label_config,
        )
    except Exception as e:
        print(f"Function [run_sample_stratification] Failed: error={e}", flush=True)

    print("\n  Sample stratification completed.")


# ==================== 6. High Contribution Patch Extraction ====================
def HighContributionPatch_Extract_all(metadata_csv, text=None):
    print("\n  This function extracts high-contribution patches from whole slide images (WSIs) based on attention scores produced by the MIL model.")

    output_dir = "./high_contri_extra"
    feature_bag_dir = "./feature_bags"
    runs_root = "./runs"
    datasplit_root = "./Data_Split"
    label_config = "config.yaml"
    sample_stratf_file = "./test_result/sample_stratification_result.csv"

    with open(label_config, "r") as f:
        config = yaml.safe_load(f)
    label_map_short = config.get("LABEL_MAP_SHORT", {})
    label_len = len(label_map_short)

    selected_subtype = []
    selected_stability = []

    print("\n  Please select subtype to extract:")
    for i in range(label_len):
        choice = input(f"  Label '{label_map_short[i]}', select? [y/n]: ").strip().lower()
        if choice in ['', 'y', 'yes']:
            selected_subtype.append(label_map_short[i])
    

    print("\n  Please select stability type:")
    choice = input(f"  stability: Consistently_Correct, select? [y/n]: ").strip().lower()
    if choice in ['', 'y', 'yes']:
        selected_stability.append("Consistently_Correct")
    choice = input(f"  stability: Highly_Variable, select? [y/n]: ").strip().lower()
    if choice in ['', 'y', 'yes']:
        selected_stability.append("Highly_Variable")
    choice = input(f"  stability: Consistently_Incorrect, select? [y/n]: ").strip().lower()
    if choice in ['', 'y', 'yes']:
        selected_stability.append("Consistently_Incorrect")
    

    topk_threshold = 0.9
    split_num = 5
    fold_num = 10

    for val in selected_subtype:
        sel_label = val
        #print(f"  Use subtype: {sel_label}")

        for val1 in selected_stability:
            sel_stability = val1
            #print(f"  Use stability: {sel_stability}")
            print(f"\n  Select subtype stability -> {sel_label} {sel_stability}")

            print("\n  Now, start work.... ")

            try:
                extract_high_contribution_patches(
                    feature_bag_dir=feature_bag_dir,
                    sample_stratf_file=sample_stratf_file,
                    output_dir=output_dir,
                    subtype=sel_label,
                    stability=sel_stability,
                    topk_threshold=topk_threshold,
                    config_path=label_config,
                    runs_root=runs_root,
                    datasplit_root=datasplit_root,
                    n_splits=split_num,
                    n_folds=fold_num,
                )
            except Exception as e:
                print(f"Function [extract_high_contribution_patches] Failed: error={e}", flush=True)

            output_dir = Path(output_dir)
            high_contri_csv = output_dir / f"{sel_label}_{sel_stability}_all.csv"
            if not high_contri_csv.exists():
                print(f"[INFO] High-contribution result file {high_contri_csv} not exists.")
                continue

            print("\n  7. High-Contribution Patches Clustering")

            hc_output_dir = "./heatmaps"
            #metadata_csv = "./Data_Split/BRCA_output_svs_file_mapping.csv"
            reps = 30
            p_item = 0.8
            n_micro = 1000

            print("\n  1) Do evaluate_clusters mode.... ")
            hc_cluster_list = [2,3,4,5,6,7,8]

            try:
                cluster_high_contribution_patches_ext(
                    metadata_csv=metadata_csv,
                    high_contri_csv=high_contri_csv,
                    disease=sel_label,
                    stability=sel_stability,
                    mode="evaluate_clusters",
                    out_dir=hc_output_dir,
                    feature_bag_dir=feature_bag_dir,
                    cluster_list=hc_cluster_list,
                    reps=reps,
                    p_item=p_item,
                    n_micro=n_micro,
                    n_clusters=0,
                )
            except Exception as e:
                print(f"Function [cluster_high_contribution_patches_ext] Failed: error={e}", flush=True)

            hc_output_dir = Path(hc_output_dir)
            hc_image_dir = Path(hc_output_dir/sel_stability/sel_label)
            if not hc_image_dir.exists():
                raise FileNotFoundError(f"Directory does not exist: {hc_image_dir}")
            hc_png_files = list(hc_image_dir.glob("*.png"))
            if len(hc_png_files) == 0:
                raise ValueError(f"No PNG files found in directory: {hc_image_dir}")

            hc_png_files = sorted(hc_png_files)

            n = len(hc_png_files)
            cols = 4
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2 * rows))
            #fig, axes = plt.subplots(rows, cols, figsize=(10, 9))
            axes = axes.flatten()

            for i in range(len(axes)):
                ax = axes[i]
                ax.axis('off')
                if i < n:
                    img = mpimg.imread(hc_png_files[i])
                    #resized_img = img[::3, ::3]
                    ax.imshow(img)
                    knum_file = hc_png_files[i].stem
                    #ax.set_title(f"{i+1}", fontsize=12)
                    ax.set_title(f"{i+1}-{knum_file}", fontsize=5)
                else:
                    ax.set_visible(False)

            plt.tight_layout()
            # figmanager = plt.get_current_fig_manager()
            # figmanager.window.state('zoomed')

            plt.show()

            print("\n  Completed. Get consensus k-selection:")
            hc_consensusk_csv = Path(hc_output_dir/sel_stability/sel_label/"consensus_k_selection.csv")
            df = pd.read_csv(hc_consensusk_csv)
            for _, row in df.iterrows():
                k_num = row["k"]
                k_score = row["cophenetic_corr"]
                print(f"   k=({k_num}) -> {k_score}")

            choice = int(input(f"\n  Please select n_clusters(2-{n+1}):"))

            print("\n  2) Do export mode.... ")
            hc_n_clusters = choice

            try:
                cluster_high_contribution_patches_ext(
                    metadata_csv=metadata_csv,
                    high_contri_csv=high_contri_csv,
                    disease=sel_label,
                    stability=sel_stability,
                    mode="export",
                    out_dir=hc_output_dir,
                    feature_bag_dir=feature_bag_dir,
                    cluster_list=hc_cluster_list,
                    reps=reps,
                    p_item=p_item,
                    n_micro=n_micro,
                    n_clusters=hc_n_clusters,
                )
            except Exception as e:
                print(f"Function [cluster_high_contribution_patches_ext] Failed: error={e}", flush=True)

    print("\n  High-contribution patch extraction and Cluster completed.")

    return


# ==================== 8. Main Functions ====================
def main():
    print("=" * 60)
    print("Construction of the Morphologic Spectrum")
    print("=" * 60)

    print("\nThis program will perform the following operations: ")
    print("  1. Data Preparation")
    print("  2. WSI Feature Extraction")
    print("  3. MIL Training")
    print("  4. MIL Testing")
    print("  5. Sample Stratification")
    print("  6. High-Contribution Patch Extraction")
    print("     High-Contribution Patches Clustering")
    print("  7. Export Morphologic Spectrum Stats")

    print("\n\nThe above functions can run in sequence.")
    choice = input("Press Enter to run all, or enter the function number to start running from that function: ").strip()

    selnum = 0
    if choice == "":
        selnum = 0
    elif choice.isdigit():
        selnum = int(choice)
    else:
        selnum = -1

    if selnum >= 0:
        output_dir = "./Data_Split"

        get_input = input("\n  Input metadata csv filename(*.csv): ").strip()
        if get_input is None:
            raise ValueError("  Must input metadata file name.")
        metafilename = f"{output_dir}/{get_input}"

        if selnum <= 1:
            print("\n1. Data Preparation")
            data_preparation_func(metafilename)
            input("\n Press enter to continue...")

        if selnum <= 2:
            print("\n2. Converting each WSI into a bag of feature vectors")
            feature_bags_func_all(metafilename)
            input("\n Press enter to continue...")

        if selnum <= 3:
            print("\n3. MIL Training")
            MIL_training_func_all()
            input("\n Press enter to continue...")

        if selnum <= 4:
            print("\n4. MIL Testing")
            MIL_testing_func_all()
            input("\n Press enter to continue...")

        if selnum <= 5:
            print("\n5. Sample Stratification")
            Sample_Stratification_func(metafilename)
            input("\n Press enter to continue...")

        if selnum <= 6:
            print("\n6. High-Contribution Patch Extraction")
            HighContributionPatch_Extract_all(metafilename)


if __name__ == "__main__":
    main()
