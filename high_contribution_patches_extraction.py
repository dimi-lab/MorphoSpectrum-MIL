import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import h5py
import yaml
import torch
import torch.nn.functional as F
from test import load_model_from_checkpoint
from train_early_stopping import get_feature_bag_path


# ---------------------
# Global constants
# ---------------------
with open("config.yaml","r") as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")
LOSS_FN = torch.nn.CrossEntropyLoss()
N_CLASSES = config["N_CLASSES"]
LABEL_MAP = config["LABEL_MAP"]
LABEL_MAP_SHORT = config["LABEL_MAP_SHORT"]
HP_INDEX = config["HP_INDEX"]
INPUT_FEATURE_SIZE = config["INPUT_FEATURE_SIZE"]


def evaluate_att_scores_slide(dataset_name, round_idx, input_feature_size, hp, feature_bag_path):

    # Load the model for the given fold, set to eval mode
    checkpoint_path =Path(f"./runs/{dataset_name}")
    model = load_model_from_checkpoint(hp, round_idx, checkpoint_path, input_feature_size, N_CLASSES,DEVICE)
    model.eval()

    #-----------------------------compute attention scores for this slide---------------------#
    att_score_slide = {}
    """
    att_score_slide = {
        "features": np.ndarray shape (N_total, 512), # patch embeddings
        "att_weight": np.ndarray shape (N_total,),   # attention weights per patch
        "att_score": np.ndarray shape (N,),          # attention scores per patch
        "subtype": 2                                 # predicted class index
    }
    """

    #Load the feature bag for this slide
    test_features,tile_ids = getCONCHFeatures(feature_bag_path)
    """
    test_features: NumPy array, shape (N, 512), N = number of patches at this scale,
                   the i-th row is the feature vector for the i-th patch.

    tile_ids:     list,
                   the i-th element is the unique patch identifier (patch_id).
    """

    att_score_slide["features"]= test_features.numpy()
    test_features=test_features.to(DEVICE)
    A, h, classifiers, Y_prob, Y_hat = predict_attention_matrix(model, test_features)
    att_score = A[Y_hat] * np.dot(h,classifiers[Y_hat]) 
    
    
    att_score_slide['subtype'] = Y_hat
    att_score_slide["att_score"]=att_score 
    att_score_slide["att_weight"]=A[Y_hat] 

    return att_score_slide, tile_ids, Y_prob, Y_hat 

def getCONCHFeatures(feature_bag_path):
    try:
        features, _,tile_ids = split_feature_tile_id(feature_bag_path) 
        print(f"feature shape:{features.shape}")
        print(f"tile ids:{len(tile_ids)}")
    except KeyError:
        raise KeyError(f"The file {feature_bag_path} doesn't contain dataset")

    features = torch.from_numpy(features)
    
    return features, tile_ids


def split_feature_tile_id(h5_path):
    with h5py.File(h5_path,"r") as file:
        features_all = file["features"][:]
        tile_ids_all_bytes = file["tile_ids"][:]
        coors_all = file["coords"][:]
        
    # transform tile_ids into python string
    tile_ids_str = [tid.decode("utf-8") for tid in tile_ids_all_bytes]
    
    #split feature vectors based on their tile level
    feats=[]
    coords=[]
    tile_ids=[]
    
    for feature,coord, tid in zip(features_all,coors_all,tile_ids_str,):
        feats.append(feature)
        coords.append(coord)
        tile_ids.append(tid)
        
    feats = np.array(feats, dtype=np.float32)

    return feats, coords, tile_ids


def predict_attention_matrix(model, feature_batch):
    with torch.no_grad():
        logits, Y_prob, Y_hat, A_raw, M, h, classifiers = model(feature_batch)
        A = F.softmax(A_raw, dim=1)
    return A.cpu().numpy(), h.cpu().numpy(), classifiers.cpu().numpy(), Y_prob.cpu().numpy(), Y_hat.item()


def process_slide(slide_id, input_feature_size, dataset_id, round_idx, hp, feature_bag_path, topk_threshold):

    # 1. Model prediction: obtain attention scores, weights, logits
    dataset_name = f"Datasplit_{dataset_id}_10_fold_by_patient_random"
    att_score_slide, tile_ids, prob, pred=evaluate_att_scores_slide(dataset_name, round_idx, input_feature_size, hp, feature_bag_path)


    pred_label = LABEL_MAP_SHORT[pred]
    print(f"\nSlide {slide_id} prediction={pred},prob={prob}\n")

    # 2. Select patches whose cumulative contribution exceeds threshold
    print(f"[INFO] Obtaining the high contribution patches...\n")
    scores = att_score_slide["att_score"]

    # print("-----scores-----")
    # for idx, val in enumerate(scores):
    #     print(f"{idx}: {val}")
    # print("\n")

    # 2.1 Sort by attention score in descending order
    sorted_indices = np.argsort(scores)[::-1]  

    print(f"sorted_indices:{sorted_indices}\n")

    sorted_scores = scores[sorted_indices]
    
    # print(f"-------sorted_scores-------")
    # for idx, val in enumerate(sorted_scores):
    #     print(f"{idx}: {val}")
    # print("\n")

    # 2.2 Compute cumulative contribution
    cumulative_scores = np.cumsum(sorted_scores)
    threshold = topk_threshold * cumulative_scores[-1]

    # 2.3 Select top-k patches
    k = np.searchsorted(cumulative_scores, threshold) + 1
    print(f"k={k}\n")

    # 2.4 Get top-k indices
    top_k_centers = sorted_indices[:k]
    top_k_ids = list(np.array(tile_ids)[top_k_centers])

    # 2.5 Extract features/weights/scores for top-k
    all_scores  = att_score_slide["att_score"]  # shape (N,)
    all_weights = att_score_slide["att_weight"] # shape (N,)
    all_feats   = att_score_slide["features"]   # shape (N, D)

    top_k_scores   = all_scores[top_k_centers]   # shape (k,)
    top_k_weights  = all_weights[top_k_centers]  # shape (k,)
    top_k_features = all_feats[top_k_centers]    # shape (k, D)
    
    return top_k_ids, top_k_scores, top_k_weights, top_k_features, pred_label

def find_models_for_slide(slide_id):
    matched_models = []
    for dataset_id in range(5):  # hard-coded: assumes 5 shuffled datasets (Datasplit_0 ... Datasplit_4)
        for round_idx in range(10):  # hard-coded: assumes 10 folds per dataset (round-0 ... round-9)
            test_slide_list = load_test_slide_list(dataset_id, round_idx)
            if slide_id in test_slide_list:
                matched_models.append((dataset_id,round_idx))
    return matched_models

def load_test_slide_list(dataset_id, round_idx):
    # hard-coded: base directory "./Data_Split" and suffix "random" in filename
    # adjust this pattern if your datasplit CSV files are stored elsewhere
    datasplit_path = f"./Data_Split/Datasplit_{dataset_id}_10_fold_by_patient_random.csv"
    datasplit_df = pd.read_csv(datasplit_path)

    test_slide_list = datasplit_df.loc[
        datasplit_df[f'round-{round_idx}'] == 'testing', 'slide_id'].tolist()

    return test_slide_list


def main(args):
    hp=HP_INDEX
    h5_base_path = args.feature_bag_dir 
    input_feature_size = INPUT_FEATURE_SIZE
    sample_strati_file_path = args.sample_stratf_file
    subtype = args.subtype
    stability = args.stability
    
    # 1) Filter target cohort by subtype + stability
    result_df = pd.read_csv(sample_strati_file_path)
    mask = (result_df['label'].astype(str).str.lower() == subtype.lower()) & \
        (result_df['stability'].astype(str).str.lower().str.replace(" ", "_").isin(
            [stability.lower()]
        ))
    subset_df = result_df.loc[mask].copy()
    if subset_df.empty:
        print(f"[INFO] No slides match subtype='{subtype}' and stability='{stability}'. Nothing to do.")
        return
    print(f"[INFO] Found {len(subset_df)} slides for subtype='{subtype}', stability='{stability}'.")


    # 2) Prepare output path
    result_base_path = Path(args.output_dir)
    result_base_path.mkdir(parents=True, exist_ok=True)
    coords_csv_path = result_base_path / f"{subtype}_{stability}_all.csv"

    header_written = False  

    # 3) Process each slide
    for _, row in subset_df.iterrows():
        slide_id = row['slide_id']
        label = row['label']
        feature_bag_path = get_feature_bag_path(h5_base_path, slide_id)

        
        print(f"\n{'='*50}")
        print(f"[INFO] Processing slide{slide_id}, (label={label})")

        #Collect results for each dataset/round
        collected_rows = []  
        model_tuples = find_models_for_slide(slide_id) # returns all (dataset_id, round_idx) where this slide was in testing
        for idx, (dataset_id, round_idx) in enumerate(model_tuples):
            print(f"[INFO] Datasplit_{dataset_id}, round_{round_idx}")

            # Obtain high-contribution patches
            top_k_ids, top_k_scores, top_k_weights, top_k_features,pred_label = process_slide(slide_id, input_feature_size, dataset_id, round_idx, hp, feature_bag_path, args.topk_threshold)

            
            for idx, center_id in enumerate(top_k_ids):
                feature = top_k_features[idx]
                feature_str = ';'.join(f"{x:.3f}" for x in feature)
                weight = top_k_weights[idx]
                atten_score = top_k_scores[idx]
                collected_rows.append([
                    slide_id, label, dataset_id, round_idx, pred_label,
                    center_id, weight, atten_score, feature_str])
            
        # Deduplicate rows
        if not collected_rows:
            print(f"[INFO] No top-k patches collected for slide {slide_id}.")
            continue

        df_slide = pd.DataFrame(
            collected_rows,
            columns=['slide_id','label','data_split','round_id','pred','tile_id','weight','score','embed']
        )
        before = len(df_slide)
        df_slide = df_slide.drop_duplicates(subset=['slide_id', 'tile_id', 'embed'], keep='first').reset_index(drop=True)
        after = len(df_slide)
        print(f"[INFO] {slide_id}: dedup {before} → {after}")

        # Write results
        if not header_written:
            df_slide.to_csv(coords_csv_path, index=False, mode='w')
            header_written = True
        else:
            df_slide.to_csv(coords_csv_path, index=False, mode='a', header=False)

        print(f"[INFO] Slide {slide_id} done; appended {after} rows to {coords_csv_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--feature_bag_dir",
        type=str,
        help="Directory containing all *_features.h5 files (feature bags).",
        required=True,
    )
    parser.add_argument(
        "--sample_stratf_file",
        type=str,
        help="CSV produced by the sample stratification step (contains 'label' and 'stability' columns).",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory where the filtered high-contribution patch CSV will be written.",
        required=True,
    )
    parser.add_argument(
        "--subtype",
        type=str,
        help="Target subtype to process.",
        required=True,
    )
    parser.add_argument(
        "--stability",
        type=str,
        choices=['Consistently_Correct', 'Consistently_Incorrect','Highly_Variable'],
        help="Target stability category",
        required=True,
    )
    parser.add_argument(
        "--topk_threshold",
        type=float,
        default=0.9,
        help="Cumulative contribution threshold for selecting high-contribution patches (0–1).",
        required=True,
    )
    args = parser.parse_args()

    main(args)

