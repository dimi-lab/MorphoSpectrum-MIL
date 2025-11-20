
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import os
import yaml
import glob
from model import AttentionNet
import numpy as np
import h5py
import random
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from train_early_stopping import FeatureBagsDataset


from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    WeightedRandomSampler,
)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score

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
HP_INDEX = config["HP_INDEX"]
INPUT_FEATURE_SIZE = config["INPUT_FEATURE_SIZE"]

plt.rcParams['axes.labelsize'] = 14  # axis label font size
plt.rcParams['xtick.labelsize'] = 14  # X-axis tick font size
plt.rcParams['ytick.labelsize'] = 14  # Y-axis tick font size
plt.rcParams['figure.dpi'] = 300      # DPI for higher resolution

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]

def define_data_sampling(test_split, workers):
    # Reproducibility of DataLoader
    g = torch.Generator()
    g.manual_seed(0)

    test_loader = DataLoader(
        dataset=test_split,
        batch_size=1,  # model expects one bag of features at the time.
        sampler=SequentialSampler(test_split),
        collate_fn=collate,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return test_loader

def evaluate_model(model, loader, n_classes, loss_fn, device):
    model.eval()

    avg_loss = 0.0

    preds = np.zeros(len(loader))
    probs = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits, Y_prob, Y_hat, _, _, _, _ = model(data)
            loss = loss_fn(logits, label) 
            avg_loss += loss.item()

            preds[batch_idx] = Y_hat.item()
            probs[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

    avg_loss /= len(loader)

    return preds, probs, labels, avg_loss

def get_class_names(df):
    n_classes = len(df["label"].unique())
    print(f"n_classes is {n_classes}")
    class_names = [None] * n_classes
    print(df["label"].unique())
    for i in df["label"].unique():
        class_names[i] = df[df["label"] == i]["class"].unique()[0]
    assert len(class_names) == n_classes
    return class_names

def calculate_sen_spec(labels, preds, n_classes):
    sensitivities=[]
    specificities=[]
    f1_scores = []
    binary_labels = label_binarize(labels, classes=list(range(n_classes)))
    binary_preds = label_binarize(preds, classes=list(range(n_classes)))
                                  
    for class_idx in range(n_classes):
        tn, fp, fn, tp = confusion_matrix(binary_labels[:, class_idx],binary_preds[:, class_idx]).ravel()

        sensitivity = tp / (tp+fn) if (tp+fn)>0 else 0
        specificity = tn / (tn+fp) if (tn+fp)>0 else 0
        
        precision = tp / (tp+fp) if (tp+fp)>0 else 0
        f1 = 2*precision*sensitivity / (precision+sensitivity) if (precision+sensitivity)>0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)
        f1_scores.append(f1)
    
    return sensitivities, specificities, f1_scores

def load_model_from_checkpoint(hp, round_idx, checkpoint_dir,input_feature_size,n_classes,device):

    test_cfg = config[hp]
    model_size = test_cfg.get("model_size")
    p_dropout_fc = float(test_cfg.get("p_dropout_fc"))
    p_dropout_atn = float(test_cfg.get("p_dropout_atn"))

    pattern = f"round_{round_idx}_{model_size}_random_{hp}*"
    hp_round_list = sorted(checkpoint_dir.glob(pattern))


    if len(hp_round_list) == 0:
        raise FileNotFoundError(
            f"[Error] No model folder found for pattern '{pattern}' in {checkpoint_dir}\n"
            f"Checked directory contents: {[p.name for p in checkpoint_dir.glob('*')]}"
        )
    if len(hp_round_list) > 1:
        raise RuntimeError(
            f"[Error] Multiple model folders match pattern '{pattern}' in {checkpoint_dir}:\n"
            + "\n".join(f" - {p}" for p in hp_round_list)
        )
    hp_round_dir_path = hp_round_list[0]

    # Find the checkpoint file inside the matched folder
    ckpt_pattern = f"round_{round_idx}*.pt"
    round_check_list = sorted(hp_round_dir_path.glob(ckpt_pattern))
    if len(round_check_list) == 0:
        raise FileNotFoundError(
            f"[Error] No checkpoint file matching '{ckpt_pattern}' in {hp_round_dir_path}\n"
            f"Folder contents: {[p.name for p in hp_round_dir_path.glob('*')]}"
        )
    if len(round_check_list) > 1:
        raise RuntimeError(
            f"[Error] Multiple checkpoint files in {hp_round_dir_path} matching '{ckpt_pattern}':\n"
            + "\n".join(f" - {p.name}" for p in round_check_list)
        )

    model_file = round_check_list[0]
    print(f"[INFO] Using checkpoint for hp={hp}, round={round_idx}: {model_file}")

    model = AttentionNet(
        model_size=model_size,
        input_feature_size=input_feature_size,
        dropout=True,
        p_dropout_fc=p_dropout_fc,
        p_dropout_atn=p_dropout_atn,
        n_classes=n_classes,
    )
    model.load_state_dict(torch.load(str(model_file),map_location=device))
    model = model.to(device)
    return model

def get_slide_level_results(hp, round_idx, labels, preds, probs, test_df):

    name_label = [LABEL_MAP[int(x)] for x in list(labels)]
    name_pred = [LABEL_MAP[int(x)] for x in list(preds)]  
    true_false=["True" if i==j else "False" for i, j in zip(labels, preds)]

    formated_probs = np.array([[f"{prob:.2f}" for prob in row] for row in probs])
    
    round_result = pd.DataFrame({
        'hp_set': [hp]*len(labels),
        'round_num': [round_idx]*len(labels),
        'slide_id': test_df['slide_id'],
        'label': list(labels),
        'name_label': name_label,
        'pred': list(preds),
        'name_pred': name_pred,
        'probs': list(formated_probs),
        'correctness': true_false
    })

    return round_result

def append_to_csv(df_to_append, round_index, slide_result_csv):
    mode="w" if round_index==0 else "a"
    header=True if round_index==0 else False

    df_to_append.to_csv(slide_result_csv, mode=mode, header=header, index=False)

def compute_round_roc_pr(labels, probs, n_classes):     
    round_fpr = {}
    round_tpr = {}
    round_AUC = {}
    round_pr = {}
    labels_binarized = label_binarize(labels, classes=list(range(n_classes)))

    for class_idx in range(n_classes):
        print(f"Computing AUC for class {class_idx}")
        fpr, tpr, _ = roc_curve(labels_binarized[:, class_idx], probs[:, class_idx])
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(labels_binarized[:, class_idx], probs[:, class_idx])

        pr_auc = average_precision_score(labels_binarized[:, class_idx], probs[:, class_idx])

        
        round_fpr[class_idx]=fpr
        round_tpr[class_idx]=tpr
        round_AUC[class_idx]=roc_auc
        round_pr[class_idx]=pr_auc
    
    return round_fpr, round_tpr, round_AUC, round_pr

def summarize_metric_per_class(all_values):
    means = {}
    stds = {}

    for class_id, values in all_values.items():
        means[class_id]=round(np.mean(values),5)
        stds[class_id]=round(np.std(values),5)

    return means, stds

def format_mean_std_dict(means, stds, decimal=4):
    formated={}
    for k in means:
        formated[k]= f"{means[k]:.{decimal}f} ± {stds[k]:.{decimal}f}"
    return formated

def plot_roc_curve_single(
        interp_tprs,
        class_all_AUC,
        mean_fpr,
        color,
        round_num,
        hp,
        save_dir,
        class_name
):
    
    mean_tpr = interp_tprs.mean(axis=0)
    std_tpr = interp_tprs.std(axis=0)

    stderr_tpr = std_tpr / np.sqrt(round_num)
    tprs_upper = np.minimum(mean_tpr + 1.96 * stderr_tpr, 1) 
    tprs_lower = np.maximum(mean_tpr - 1.96 * stderr_tpr, 0) 

    mean_auc = np.mean(class_all_AUC)
    std_auc = np.std(class_all_AUC)

    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color=color, label=f'Mean ROC')
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=0.2, label='95% Confidence Interval')

    auc_text = f'AUC = {mean_auc:.2f} ± {std_auc:.2f}'
    plt.text(0.55, 0.14, auc_text, fontsize=18, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.55, 0.23, hp, fontsize=18, bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{hp} {class_name} ROC Curve (95% Confidence Interval)')
    plt.legend(loc='lower right')

    # 保存图像
    save_path = save_dir/f"{hp}_roc_curve_{class_name}.png"
    plt.savefig(save_path)
    plt.close()
    
def plot_roc_curves_all(
        all_fpr,
        all_tpr,
        round_num,
        color_list,
        save_dir,
        hp,
        class_names
):

    plt.figure()
    mean_fpr = np.linspace(0, 1, 100) 
    macro_tprs =[]

    for class_id, cname in enumerate(class_names):
        interp_tprs=[]
        for i in range(round_num):
            interp_tpr = np.interp(mean_fpr, all_fpr[class_id][i], all_tpr[class_id][i])
            interp_tpr[0] = 0
            interp_tprs.append(interp_tpr)
        
        interp_tprs = np.array(interp_tprs)
        mean_tpr = interp_tprs.mean(axis=0)

        plt.plot(mean_fpr,mean_tpr,color=color_list[class_id],label=cname)
        macro_tprs.append(mean_tpr)

    macro_tprs = np.array(macro_tprs)
    macro_tpr = macro_tprs.mean(axis=0)
    plt.plot(mean_fpr, macro_tpr, color='black', linestyle="dashed", label="Macro-average")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    save_path = save_dir/f"{hp}_macro_averaged_roc_curve.png"
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix_from_csv(csv_path, save_dir):

    df = pd.read_csv(csv_path)
    
    y_true = df['label'].astype(int)
    y_pred = df['pred'].astype(int)
    
    labels_order = sorted(list(LABEL_MAP.keys())) if isinstance(list(LABEL_MAP.keys())[0], int) else sorted(df['label'].unique().astype(int).tolist())
    cm = confusion_matrix(y_true, y_pred,labels=labels_order)
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_order, yticklabels=labels_order)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
    
    save_path = save_dir/f"confusion_matrix.png"
    plt.savefig(save_path)
    plt.close()

def random_baseline_by_distribution_f1(y_true, classes, n_trials=50):
    """
    {
        class_id1: random_f1
        class_id2: random_f1
        ...
    }
    """ 

    counts = np.array([np.sum(y_true==c) for c in classes])
    probs = counts/counts.sum()
    
    trial_scores = []
    for _ in range(n_trials):
        y_pred = np.random.choice(classes, size=len(y_true), p=probs)
        score = f1_score(y_true, y_pred, labels=classes, average=None)
        trial_scores.append(score)
    trial_scores = np.array(trial_scores)
    mean_scores = np.mean(trial_scores,axis=0)
    result = {c:s for c,s in zip(classes,mean_scores)}

    return result

def main(args):
    round_num = args.round_num
    test_set_path = args.manifest
    dataset_name = test_set_path.split('/')[-1].split('.csv')[0]
    input_feature_size = INPUT_FEATURE_SIZE

    
    #get checkpoint path
    checkpoint_path = Path(args.checkpoint_dir) / dataset_name

    roc_path = Path(f"./ROC_Curve/{dataset_name}")
    roc_path.mkdir(parents=True, exist_ok=True)

    final_result_path = Path(f"./test_result/{dataset_name}")
    final_result_path.mkdir(parents=True, exist_ok=True)

    each_slide_result = Path(f"./test_result/{dataset_name}/each_slide_result")
    each_slide_result.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(test_set_path)

    # ------------------------- Testing Loop for each hp -------------------------
    hp = HP_INDEX
    print(f"[INFO] Start testing model for hyperparameter set {hp}")
    
    # 初始化存储容器
    all_fpr = {class_idx: [] for class_idx in range(N_CLASSES)} 
    all_tpr = {class_idx: [] for class_idx in range(N_CLASSES)}
    all_AUC = {class_idx: [] for class_idx in range(N_CLASSES)}
    all_sen = {class_idx: [] for class_idx in range(N_CLASSES)}
    all_spe = {class_idx: [] for class_idx in range(N_CLASSES)}
    all_pr = {class_idx: [] for class_idx in range(N_CLASSES)}
    all_f1 = {class_idx: [] for class_idx in range(N_CLASSES)}
    all_random_f1 ={class_idx: [] for class_idx in range(N_CLASSES)} # store random guess baseline
    
    # ------------------------- Fold Loop -------------------------
    for round_idx in range(round_num): 

        print(f"[INFO] Testing hp set {hp}, round {round_idx}")

        #load model
        model = load_model_from_checkpoint(hp, round_idx, checkpoint_path, input_feature_size,N_CLASSES,DEVICE)

        # get test set for this round and convert to feature bags
        test_set = df[df[f'round-{round_idx}']=='testing']
        test_split = FeatureBagsDataset(test_set,args.feature_bag_dir)

        test_loader = define_data_sampling(
                    test_split,
                    workers=args.workers,
                )
        class_names = get_class_names(test_set)

        # run model and evaluate predictions
        preds, probs, labels, test_loss = evaluate_model(
                        model, test_loader, N_CLASSES, LOSS_FN, DEVICE
                    )
        print(f"preds is {preds}\n")

        print(f"label is {labels}\n")
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        label_distribution = dict(zip(unique_labels.astype(int), label_counts))

        print("Label distribution (count per class):", label_distribution)

        np.set_printoptions(suppress=True, precision=4, floatmode='fixed')

        # compute random baseline F1
        random_f1s=random_baseline_by_distribution_f1(labels, np.unique(labels))

        #compute sensitivity/specificity/F1
        sens, specs, f1s = calculate_sen_spec(labels, preds, N_CLASSES)
        print(f"sensitiviteis:{sens}")
        print(f"specificities:{specs}")
        print(f"F1 scores:{f1s}")
        for class_idx in range(N_CLASSES):
            all_sen[class_idx].append(sens[class_idx])
            all_spe[class_idx].append(specs[class_idx])
            all_f1[class_idx].append(f1s[class_idx])
            all_random_f1[class_idx].append(random_f1s[class_idx])
        

        #compute ROC and PR per class
        round_fpr, round_tpr, round_AUC, round_pr=compute_round_roc_pr(labels, probs, N_CLASSES)
        for c in range(N_CLASSES):
            all_fpr[c].append(round_fpr[c])
            all_tpr[c].append(round_tpr[c])
            all_AUC[c].append(round_AUC[c])
            all_pr[c].append(round_pr[c])

        #obtain each slide prediction
        round_result = get_slide_level_results(hp, round_idx, labels, preds, probs, test_set)
        slide_csv_path=each_slide_result/f"{hp}_each_slide_result.csv"
        append_to_csv(round_result, round_idx, slide_csv_path)
        # ------------------------- Fold Loop Ends -------------------------

    # ---------------------Summarize results----------------------------
    
    
    #---------write per-round CSV--------
    round_rows = []
    for class_idx in range(N_CLASSES):
        for round_idx in range(round_num):
            round_rows.append({
            'hp_set': hp,
            'subtype': LABEL_MAP[class_idx],
            'round': round_idx,
            'AUC': f"{all_AUC[class_idx][round_idx]:.4f}",
            'Sensitivity': f"{all_sen[class_idx][round_idx]:.4f}",
            'Specificity': f"{all_spe[class_idx][round_idx]:.4f}",
            'PR_AUC': f"{all_pr[class_idx][round_idx]:.4f}",
            'F1': f"{all_f1[class_idx][round_idx]:.4f}",
        })
    round_df_summary = pd.DataFrame(round_rows)
    round_csv_path=final_result_path/f"each_round_result.csv"
    mode = 'w' 
    header = True 
    round_df_summary.to_csv(round_csv_path, mode=mode, header=header, index=False)

    print(f"[INFO] {hp} 10-round results written to {round_csv_path}")
    
    #---------write averaged metrics to final_result.csv--------
    mean_sen, std_sen = summarize_metric_per_class(all_sen)
    mean_spe, std_spe = summarize_metric_per_class(all_spe)
    mean_f1, std_f1 = summarize_metric_per_class(all_f1)
    mean_pr_auc, std_pr_auc = summarize_metric_per_class(all_pr)
    mean_AUC, std_AUC = summarize_metric_per_class(all_AUC)
    mean_random_f1, std_random_f1 = summarize_metric_per_class(all_random_f1)
    macro_auc = np.mean(list(mean_AUC.values()))
    macro_auc_std = np.std(list(mean_AUC.values()))
    

    rows = []
    
    for c_idx, cname in enumerate(class_names):
        rows.append({
            'hp_set': hp,
            'subtype': cname,
            'mean_test_auc': mean_AUC[c_idx],
            'mean_test_auc_std': std_AUC[c_idx],
            'macro_average_AUC': f"{macro_auc:.4f}",
            'macro_average_AUC_std': f"{macro_auc_std:.4f}",
            'mean_test_sensitivity': mean_sen[c_idx],
            'mean_test_sensitivity_std': std_sen[c_idx],
            'mean_test_specificity': mean_spe[c_idx],
            'mean_test_specificity_std': std_spe[c_idx],
            'mean_pr_auc': mean_pr_auc[c_idx],
            'mean_pr_auc_std': std_pr_auc[c_idx],
            'mean_f1': mean_f1[c_idx],
            'mean_f1_std': std_f1[c_idx],
            'mean_random_f1': mean_random_f1[c_idx],
            'mean_random_f1_std': std_random_f1[c_idx],
        })

    df_summary = pd.DataFrame(rows)

    final_csv_path=final_result_path/f"final_result.csv"
    mode = 'w' 
    header = True 
    df_summary.to_csv(final_csv_path, mode=mode, header=header, index=False)

    print(f"[INFO] {hp} averaged results written to {final_csv_path}")
    # ------------------------- AUC Figure plot begins -------------------------
    print(f"[INFO] Plotting ROC curves for {hp}")
    mean_fpr = np.linspace(0, 1, 100)
    color_list=['green','blue','darkorange','red']

    #per-class ROC with 95% CI
    for class_idx, cname in enumerate(class_names):
        interp_tprs=[]
        for i in range(round_num):
            interp_tpr = np.interp(mean_fpr, all_fpr[class_idx][i], all_tpr[class_idx][i])
            interp_tpr[0]=0
            interp_tprs.append(interp_tpr)
        interp_tprs = np.array(interp_tprs)

        plot_roc_curve_single(interp_tprs, all_AUC[class_idx],mean_fpr,color_list[class_idx],round_num,hp,roc_path,cname)

    
    plot_roc_curves_all(all_fpr, all_tpr, round_num, color_list, roc_path, hp, class_names)

    # -------------------------plot confusion matrix-------------------------
    print(f"[INFO] Plotting confusion matrix for {hp}")
    plot_confusion_matrix_from_csv(slide_csv_path, final_result_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing Script")
    parser.add_argument(
        "--manifest",
        type=str,
        help="CSV file listing all slides, their labels, and which split (train/test/val) they belong to.",
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="./runs"
    )
    parser.add_argument(
        "--feature_bag_dir",
        type=str,
        help="Directory where all *_features.h5 files are stored",
    )
    parser.add_argument(
        "--round_num",
        type=int,
        help="number of rounds for testing"
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loaders.",
        type=int,
        default=4,
    )
    args = parser.parse_args()
    main(args)