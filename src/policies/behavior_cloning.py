import json
import random
import uuid

from pathlib import Path
from datetime import datetime, timezone

import yaml
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# MODEL
# ============================================================

class BehaviorCloningMLP(nn.Module):

    def __init__(self, config):

        super().__init__()

        layers = []

        input_dim = config["input_dim"]

        for hidden_dim in config["hidden_dims"]:

            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])

            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        self.movement_head = nn.Linear(
            input_dim,
            config["movement_output_dim"]
        )

        self.click_head = nn.Linear(
            input_dim,
            config["click_output_dim"]
        )

    def forward(self, observations):

        features = self.backbone(observations)

        movement = self.movement_head(features)

        click_logits = self.click_head(features)

        return movement, click_logits


# ============================================================
# REPRODUCIBILITY AND DEVICE
# ============================================================

def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device):

    if device == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(device)


def build_model(config, device):

    if config["type"] != "mlp":
        raise ValueError("Unsupported model type.")

    if config["activation"] != "relu":
        raise ValueError("Unsupported activation.")

    return BehaviorCloningMLP(config).to(device)


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# DATA LOADING
# ============================================================

def build_data_loader(
    data,
    batch_size,
    shuffle=False,
    seed=42
):

    observations = torch.tensor(
        data["observations"],
        dtype=torch.float32
    )

    movement = torch.tensor(
        data["actions"][:, :2],
        dtype=torch.float32
    )

    clicks = torch.tensor(
        data["actions"][:, 2:3],
        dtype=torch.float32
    )

    dataset = TensorDataset(
        observations,
        movement,
        clicks
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator
    )


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def build_loss_functions(config, device):

    if config["movement"] != "mse":
        raise ValueError("Unsupported movement loss.")

    if config["click"] != "bce":
        raise ValueError("Unsupported click loss.")

    return {
        "movement": nn.MSELoss(),

        "click": nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [config["click_positive_weight"]],
                dtype=torch.float32,
                device=device
            )
        )
    }


def compute_losses(
    movement_prediction,
    click_logits,
    movement_target,
    click_target,
    losses,
    config
):

    movement_loss = losses["movement"](
        movement_prediction,
        movement_target
    )

    click_loss = losses["click"](
        click_logits,
        click_target
    )

    total_loss = (
        config["movement_weight"] * movement_loss
        + config["click_weight"] * click_loss
    )

    return total_loss, movement_loss, click_loss


# ============================================================
# SINGLE TRAINING / VALIDATION EPOCH
# ============================================================

def execute_epoch(
    model,
    data_loader,
    losses,
    loss_config,
    device,
    click_threshold=0.5,
    optimizer=None
):

    training = optimizer is not None

    model.train(training)

    totals = {
        "loss": 0.0,
        "movement_loss": 0.0,
        "click_loss": 0.0
    }

    sample_count = 0

    tp = fp = tn = fn = 0

    with torch.set_grad_enabled(training):

        for observations, movement, clicks in data_loader:

            observations = observations.to(device)
            movement = movement.to(device)
            clicks = clicks.to(device)

            if training:
                optimizer.zero_grad()

            movement_pred, click_logits = model(
                observations
            )

            total_loss, movement_loss, click_loss = (
                compute_losses(
                    movement_pred,
                    click_logits,
                    movement,
                    clicks,
                    losses,
                    loss_config
                )
            )

            if not torch.isfinite(total_loss):
                raise RuntimeError("Non-finite training loss.")

            if training:

                total_loss.backward()
                optimizer.step()

            count = len(observations)

            sample_count += count

            totals["loss"] += total_loss.item() * count

            totals["movement_loss"] += (
                movement_loss.item() * count
            )

            totals["click_loss"] += (
                click_loss.item() * count
            )

            predicted = (
                torch.sigmoid(click_logits.detach())
                >= click_threshold
            )

            actual = clicks.bool()

            tp += int((predicted & actual).sum().item())
            fp += int((predicted & ~actual).sum().item())
            tn += int((~predicted & ~actual).sum().item())
            fn += int((~predicted & actual).sum().item())

    if sample_count == 0:
        raise ValueError("Empty data loader.")

    return {
        **{
            name: value / sample_count
            for name, value in totals.items()
        },

        "sample_count": sample_count,

        "click_true_positives": tp,
        "click_false_positives": fp,
        "click_true_negatives": tn,
        "click_false_negatives": fn,

        "click_precision": (
            tp / (tp + fp)
            if tp + fp else None
        ),

        "click_recall": (
            tp / (tp + fn)
            if tp + fn else None
        ),

        "actual_click_rate": (
            (tp + fn) / sample_count
        ),

        "predicted_click_rate": (
            (tp + fp) / sample_count
        )
    }


# ============================================================
# RUN MANAGEMENT
# ============================================================

def create_run(experiment_dir, config, dataset_manifest):

    experiment_dir = Path(experiment_dir)

    run_id = f"run_{uuid.uuid4().hex[:12]}"

    run_dir = experiment_dir / "runs" / run_id

    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "checkpoints").mkdir()
    (run_dir / "evaluations").mkdir()

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )

    (run_dir / "dataset_manifest.yaml").write_text(
        yaml.safe_dump(
            dataset_manifest,
            sort_keys=False
        )
    )

    metadata = {
        "run_id": run_id,

        "dataset_id":
            dataset_manifest["dataset"]["id"],

        "experiment_id":
            config["experiment"]["id"],

        "status": "running",

        "started_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "ended_at": None
    }

    save_run_metadata(run_dir, metadata)

    return run_dir, metadata


def save_run_metadata(run_dir, metadata):

    (Path(run_dir) / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            allow_nan=False
        )
    )


def save_training_history(run_dir, history):

    path = Path(run_dir) / "training_history.json"

    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(
            history,
            indent=2,
            allow_nan=False
        )
    )

    temporary.replace(path)


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    config,
    metrics
):

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "model_config":
            config["model"],

        "inference_config":
            config["inference"],

        "metrics": metrics
    }

    torch.save(checkpoint, path)


def load_checkpoint(path, device="cpu"):

    device = torch.device(device)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    model = build_model(
        checkpoint["model_config"],
        device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(
    model,
    train_loader,
    validation_loader,
    config,
    device,
    run_dir
):

    training_config = config["training"]
    checkpoint_config = config["checkpoint"]

    if training_config["optimizer"] != "adam":
        raise ValueError("Unsupported optimizer.")

    if checkpoint_config["monitor"] != "validation_loss":
        raise ValueError("Unsupported checkpoint monitor.")

    if training_config["epochs"] < 1:
        raise ValueError("At least one epoch is required.")

    if checkpoint_config["save_every_epochs"] < 1:
        raise ValueError("Invalid checkpoint interval.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"]
    )

    losses = build_loss_functions(
        config["loss"],
        device
    )

    history = []

    best_loss = float("inf")
    best_epoch = None

    checkpoint_dir = Path(run_dir) / "checkpoints"

    for epoch in range(
        1,
        training_config["epochs"] + 1
    ):

        train_metrics = execute_epoch(
            model,
            train_loader,
            losses,
            config["loss"],
            device,
            config["inference"]["click_threshold"],
            optimizer=optimizer
        )

        validation_metrics = execute_epoch(
            model,
            validation_loader,
            losses,
            config["loss"],
            device,
            config["inference"]["click_threshold"]
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics
        }

        history.append(record)

        save_training_history(
            run_dir,
            history
        )

        validation_loss = validation_metrics["loss"]

        if validation_loss < best_loss:

            best_loss = validation_loss
            best_epoch = epoch

            if checkpoint_config["save_best"]:

                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    config,
                    record
                )

        if epoch % checkpoint_config[
            "save_every_epochs"
        ] == 0:

            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                config,
                record
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_metrics['loss']:.5f} | "
            f"val={validation_loss:.5f} | "
            f"movement={validation_metrics['movement_loss']:.5f} | "
            f"click={validation_metrics['click_loss']:.5f} | "
            f"clicks_predicted="
            f"{validation_metrics['click_true_positives'] + validation_metrics['click_false_positives']}"
        )

    save_checkpoint(
        checkpoint_dir / "final.pt",
        model,
        optimizer,
        epoch,
        config,
        history[-1]
    )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss
    }


# ============================================================
# POLICY INFERENCE
# ============================================================

@torch.no_grad()
def predict_action(
    model,
    observation,
    click_threshold=0.5
):

    model.eval()

    device = next(model.parameters()).device

    observation = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=device
    ).reshape(1, -1)

    movement, click_logits = model(observation)

    probability = float(
        torch.sigmoid(click_logits).item()
    )

    dx, dy = movement[0].cpu().tolist()

    return {
        "delta_x": dx,
        "delta_y": dy,
        "click_probability": probability,
        "click": probability >= click_threshold
    }