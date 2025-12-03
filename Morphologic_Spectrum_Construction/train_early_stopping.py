import itertools
import os
import yaml
import random
import subprocess
import time
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_sample_weight
import sys
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    WeightedRandomSampler,
)
from torch.utils.tensorboard import SummaryWriter

try:
    from .model import AttentionNet  
except ImportError:
    from model import AttentionNet   

def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]


def get_feature_bag_path(data_dir, slide_id):
    fea_bag_dir = Path(data_dir)
    matching_file = list(fea_bag_dir.glob(f"{slide_id}*_features.h5"))
    if not matching_file:
        raise FileNotFoundError(
            f"No feature bag file found for slide_id={slide_id} in {fea_bag_dir}"
        )
    fea_bag_file=str(matching_file[0])
    #print(f"Found feature bag file:{fea_bag_file}")
    return fea_bag_file

class FeatureBagsDataset(Dataset):
    def __init__(self, df, data_dir):
        self.slide_df = df.copy().reset_index(drop=True)
        self.data_dir = data_dir

    def __getitem__(self, idx):
        slide_id = self.slide_df["slide_id"][idx]
        label = self.slide_df["label"][idx]

        full_path = get_feature_bag_path(self.data_dir, slide_id)
        try:
            with h5py.File(full_path, "r") as hdf5_file:
                try:
                    features = hdf5_file["features"][:]
                except KeyError:
                    raise KeyError(f"'features' dataset not found in H5 file: {full_path}")

                try:
                    coords = hdf5_file["coords"][:]
                except KeyError:
                    raise KeyError(f"'coords' dataset not found in H5 file: {full_path}")

        except OSError as e:
            raise OSError(f"Failed to open H5 file: {full_path}\nOriginal error: {e}")
        features = torch.from_numpy(features)
        return features, label, coords

    def __len__(self):
        return len(self.slide_df)


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

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, min_epochs=50, verbose=False):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_epochs (int): Earliest epoch possible for stopping.
            verbose (bool): If True, prints messages for e.g. each validation loss improvement.
        """
        self.patience = patience
        self.min_epochs = min_epochs
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, log_dir,ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.min_epochs:
                self.early_stop = True
        else:
            self.best_score = score
            for ckpt in glob.glob(f"{log_dir}/*.pt"):
                os.remove(ckpt)
                print(f"{ckpt} has been removed")
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def compute_auc(labels, probs):
    assert probs.shape[0] > 0
    assert probs.shape[1] > 1

    if probs.shape[1] == 2:
        raise Exception(
            "If you are doing binary classification, make sure to revisit the #applicability of AUC macro-averaging."
        )
    return roc_auc_score(labels, probs, multi_class="ovr", average="macro")


def compute_auc_each_class(labels, probs):
    # Per-class AUC in a multi-class context.
    assert probs.shape[0] > 0
    assert (
        probs.shape[1] > 2
    ), "This function is only relevant for multi-class (non-binary) tasks."
    return [roc_auc_score(labels == i, probs[:, i]) for i in range(probs.shape[1])]


def render_confusion_matrix(cm, class_names, normalize=False):
    """Render confusion matrix as a matplotlib figure."""
    title = "Confusion matrix"
    cmap = plt.cm.Blues
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    figure = plt.figure(figsize=(8, 8))
    vmax = 1 if normalize else None
    plt.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    return figure


def run_train_eval_loop(
    train_loader,
    val_loader,
    input_feature_size,
    class_names,
    hparams,
    run_id,
    full_training,
    save_checkpoints,
    dataset_name,
    round_id,
):
    # ========= CUSTOMIZATION POINT START =========
    writer = SummaryWriter(os.path.join(f"./runs/{dataset_name}", run_id))

    device = torch.device("cuda")

    gpu_visible_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    print(f"gpu_visible_str is {gpu_visible_str}")


    loss_fn = torch.nn.CrossEntropyLoss()#交叉熵损失函数
    n_classes = len(class_names)

    
    model = AttentionNet(
        model_size=hparams["model_size"],
        input_feature_size=input_feature_size,
        dropout=True,
        p_dropout_fc=hparams["p_dropout_fc"],
        p_dropout_atn=hparams["p_dropout_atn"],
        n_classes=n_classes,
    )
    model.to(device) 
    print(model)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_trainable_params} parameters")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hparams["initial_lr"],
        weight_decay=hparams["weight_decay"],
    )

    # Using a multi-step LR decay routine.
    milestones = [int(x) for x in hparams["milestones"].split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=hparams["gamma_lr"]
    )

    early_stop_tracker = EarlyStopping(
        patience=hparams["earlystop_patience"], 
        min_epochs=hparams["earlystop_min_epochs"],
        verbose=True,
    )

    metric_history = []

    for epoch in range(hparams["max_epochs"]):
        model.train()
        epoch_start_time = time.time()
        train_loss = 0.0
        preds = np.zeros(len(train_loader))
        probs = np.zeros((len(train_loader), n_classes))
        labels = np.zeros(len(train_loader))

        batch_start_time = time.time()
        for batch_idx, (data, label) in enumerate(train_loader):
            data_load_duration = time.time() - batch_start_time
            data, label = data.to(device), label.to(device) 
            logits, Y_prob, Y_hat, _, _, _, _ = model(data) 
            
            preds[batch_idx] = Y_hat.item() 
            probs[batch_idx] = Y_prob.cpu().detach().numpy() 
            labels[batch_idx] = label.item() 
            loss = loss_fn(logits, label)
            train_loss += loss.item()

            # backward pass
            loss.backward() 

            # step
            optimizer.step() 
            optimizer.zero_grad() 

            batch_duration = time.time() - batch_start_time
            batch_start_time = time.time()

            print(
                f"epoch {epoch}, batch {batch_idx}, batch took: {batch_duration:.2f}s, data loading: {data_load_duration:.2f}s, loss: {loss.item():.4f}, label: {label.item()}"
            )
            
            writer.add_scalar("data_load_duration", data_load_duration, epoch)
            writer.add_scalar("batch_duration", batch_duration, epoch)

        epoch_duration = time.time() - epoch_start_time
        print(f"Finished training on epoch {epoch} in {epoch_duration:.2f}s")

        
        train_loss /= len(train_loader)
        train_avg_auc = compute_auc(labels, probs)

        writer.add_scalar("epoch_duration", epoch_duration, epoch)
        writer.add_scalar("LR", get_lr(optimizer), epoch)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("AUC/train", train_avg_auc, epoch)

        if n_classes > 2:
            train_single_aucs = compute_auc_each_class(labels, probs)
            for class_index in range(n_classes):
                writer.add_scalar(
                    f"AUC/train-{class_names[class_index]}",
                    train_single_aucs[class_index],
                    epoch,
                )

        for class_index in range(n_classes):
            writer.add_pr_curve(
                f"PRcurve/train-{class_names[class_index]}",
                labels == class_index,
                probs[:, class_index],
                epoch,
            )

        if not full_training:
            print("Evaluating model on validation set...")

            preds, probs, labels, val_loss = evaluate_model(
                model, val_loader, n_classes, loss_fn, device
            )

            val_avg_auc = compute_auc(labels, probs)

            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("AUC/validation", val_avg_auc, epoch)

            for class_index in range(n_classes):
                writer.add_pr_curve(
                    f"PRcurve/validation-{class_names[class_index]}",
                    labels == class_index,
                    probs[:, class_index],
                    epoch,
                )
            metric_dict = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_auc": val_avg_auc,
                "trainable_params": n_trainable_params,
            }

            if n_classes > 2:
                val_single_aucs = compute_auc_each_class(labels, probs)
                for class_index in range(n_classes):
                    writer.add_scalar(
                        f"AUC/validation-{class_names[class_index]}",
                        val_single_aucs[class_index],
                        epoch,
                    )
                for idx, each_auc_class in enumerate(val_single_aucs):
                    metric_dict[f"val_auc_{class_names[idx]}"] = each_auc_class

                cm = confusion_matrix(
                    [class_names[l] for l in labels.astype(int)],
                    [class_names[p] for p in preds.astype(int)],
                    labels=class_names,
                )
                
            
            metric_history.append(metric_dict)
            
            early_stop_tracker(epoch, val_loss, model, writer.log_dir,ckpt_name = os.path.join(writer.log_dir, f"round_{round_id}_epoch_{epoch}_checkpoint.pt"))
            

        if save_checkpoints:
            torch.save(
                model.state_dict(),
                os.path.join(writer.log_dir, f"{epoch}_checkpoint.pt"),
            )

        # Update LR decay.
        scheduler.step()

        if early_stop_tracker.early_stop:
            print(
                f"Early stop criterion reached. Broke off training loop after epoch {epoch}."
            )
            break

    if not full_training:
        # Log the hyperparameters of this experiment and the performance metrics of the best epoch.
        best = sorted(metric_history, key=lambda x: x["val_loss"])[0]
        writer.add_hparams(hparams, best)

    writer.close()


def define_data_sampling(train_split, val_split, method, workers):
    # Reproducibility of DataLoader
    g = torch.Generator()
    g.manual_seed(0)

    # Set up training data sampler.
    if method == "random":
        print("random sampling setting")
        train_loader = DataLoader(
            dataset=train_split,
            batch_size=1,  # model expects one bag of features at the time.
            shuffle=True,
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    elif method == "balanced":
        print("balanced sampling setting")
        train_labels = train_split.slide_df["label"]

        # Compute sample weights to alleviate class imbalance with weighted sampling.
        sample_weights = compute_sample_weight("balanced", train_labels)

        train_loader = DataLoader(
            dataset=train_split,
            batch_size=1,  # model expects one bag of features at the time.
            # Use the weighted sampler using the precomputed sample weights.
            # Note that replacement is true by default, so
            # some slides of rare classes will be sampled multiple times per epoch.
            sampler=WeightedRandomSampler(sample_weights, len(sample_weights)),
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        raise Exception(f"Sampling method '{method}' not implemented.")

    # val_split would be an empty list if not validation is asked in training.
    if len(val_split) == 0:
        val_loader = val_split
    else:
        val_loader = DataLoader(
            dataset=val_split,
            batch_size=1,  # model expects one bag of features at the time.
            sampler=SequentialSampler(val_split),
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    return train_loader, val_loader


def get_class_names(df):
    n_classes = len(df["label"].unique())
    class_names = [None] * n_classes
    for i in df["label"].unique():
        class_names[i] = df[df["label"] == i]["class"].unique()[0]
    assert len(class_names) == n_classes
    return class_names


from pathlib import Path

def train_single_round(
    manifest,
    feature_bag_dir,
    round_idx: int,
    workers: int = 4,
    full_training_index: int | None = None,
    config_path: str | Path | None = None,
):
    """
    Train CLAM for a single CV round (or on the full dataset).
    """

    # ---------- 1. Resolve paths and load the configuration file ----------
    manifest = Path(manifest)
    feature_bag_dir = Path(feature_bag_dir)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    hp_index = config["HP_INDEX"]
    input_feature_size = config["INPUT_FEATURE_SIZE"]

    # ---------- 2. Set random seeds ----------
    set_seed()

    if not torch.cuda.is_available():
        raise Exception(
            "No CUDA device available. Training without one is not feasible."
        )

    # ---------- 3. Load manifest ----------
    df = pd.read_csv(manifest)
    class_names = get_class_names(df)
    round_index_str = str(round_idx)

    dataset_name = manifest.stem  # e.g. Datasplit_0_10_fold_by_patient

    # ---------- 4. Decide whether to perform full training or cross-validation ----------
    if full_training_index is not None:
        print(
            f"Training on full dataset (training + validation) with hparam set {full_training_index}"
        )
        training_set = df
        val_split = [None]
        base_run_id = "full_dataset"
    else:
        print(f"=> Round {round_index_str}")
        base_run_id = f"round_{round_index_str}"
        try:
            training_set = df[df[f"round-{round_index_str}"] == "training"]
            validation_set = df[df[f"round-{round_index_str}"] == "validation"]
        except KeyError:
            raise Exception(
                f"Column round-{round_index_str} does not exist in {manifest}"
            )
        val_split = FeatureBagsDataset(validation_set, feature_bag_dir)

    train_split = FeatureBagsDataset(training_set, feature_bag_dir)

    # ---------- 5. Git SHA & basic logging ----------
    try:
        commit = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
    except subprocess.CalledProcessError:
        commit = "unknown"

    git_sha = commit
    train_run_id = f"{git_sha}_{time.strftime('%Y%m%d-%H%M')}"

    print(f"=> Git SHA {train_run_id}")
    print(f"=> Training on {len(train_split)} samples")
    print(f"=> Validating on {len(val_split)} samples")

    # ---------- 6. Hyperparameters from config ----------
    train_cfg = config[hp_index]
    hparams = dict(
        sampling_method=train_cfg["sampling_method"],
        max_epochs=train_cfg["max_epochs"],
        # Early stopping
        earlystop_patience=train_cfg["earlystop_patience"],
        earlystop_min_epochs=train_cfg["earlystop_min_epochs"],
        # Optimizer
        initial_lr=float(train_cfg["initial_lr"]),
        milestones=train_cfg["milestones"],
        gamma_lr=train_cfg["gamma_lr"],
        weight_decay=float(train_cfg["weight_decay"]),
        # Model architecture
        model_size=train_cfg["model_size"],
        p_dropout_fc=train_cfg["p_dropout_fc"],
        p_dropout_atn=train_cfg["p_dropout_atn"],
    )
    hparam_sets = [hparams]
    hparams_to_use = hparam_sets
    if full_training_index is not None:
        hparams_to_use = [hparam_sets[full_training_index]]

    print(f"Start Training and Validation on {dataset_name}")

    # ---------- 7. training loop ----------
    for i, hps in enumerate(hparams_to_use):
        run_id = (
            f"{base_run_id}_{hps['model_size']}_"
            f"{hps['sampling_method']}_{hp_index}_{train_run_id}"
        )
        print(f"Running train-eval loop {i} for {run_id}")
        print(hps)

        train_loader, val_loader = define_data_sampling(
            train_split,
            val_split,
            method=hps["sampling_method"],
            workers=workers,
        )

        run_train_eval_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            input_feature_size=input_feature_size,
            class_names=class_names,
            hparams=hps,
            run_id=run_id,
            full_training=full_training_index is not None,
            save_checkpoints=full_training_index is not None,
            dataset_name=dataset_name,
            round_id=round_index_str,
        )

    print("Finished training.")



def main(args):
    train_single_round(
        manifest=args.manifest,
        feature_bag_dir=args.feature_bag_dir,
        round_idx=args.round,
        workers=args.workers,
        full_training_index=args.full_training,
        config_path=args.config,  
    )

    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--manifest",
        type=str,
        help="CSV file listing all slides, their labels, and which split (train/test/val) they belong to.",
        required=True,
    )
    parser.add_argument(
        "--round",
        type=int,
        help="Index of the round in cross-validation",
        required=True,
    )
    parser.add_argument(
        "--feature_bag_dir",
        type=str,
        help="Directory where all *_features.h5 files are stored",
        required=True,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (e.g. Morphologic_Spectrum_Construction/config.yaml). "
            "If not provided, defaults to config.yaml next to this script.",
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loaders.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--full_training",
        type=int,
        help="Provide an index of the hyperparameter set you want to use to train the final model on the combined training and validation sets.",
    )
    args = parser.parse_args()

    main(args)
