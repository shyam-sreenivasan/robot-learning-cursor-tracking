import argparse
import time

from pathlib import Path
from datetime import datetime, timezone

import yaml
import torch

from src.preprocessing.preprocessing import (
    load_dataset,
    load_dataset_manifest
)

from src.policies.behavior_cloning import (
    set_random_seed,
    resolve_device,
    build_model,
    count_parameters,
    build_data_loader,
    create_run,
    train_model,
    save_run_metadata
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="dataset_001"
    )

    parser.add_argument(
        "--experiment",
        default="mlp_baseline"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    args = parser.parse_args()

    dataset_dir = (
        Path("data/processed") / args.dataset
    )

    experiment_dir = (
        dataset_dir
        / "experiments"
        / args.experiment
    )

    config = yaml.safe_load(
        (experiment_dir / "config.yaml").read_text()
    )

    if args.seed is not None:
        config["training"]["random_seed"] = args.seed

    seed = config["training"]["random_seed"]

    set_random_seed(seed)

    device = resolve_device(
        config["training"]["device"]
    )

    # 1. Load dataset.

    dataset_manifest = load_dataset_manifest(
        dataset_dir
    )

    train_data = load_dataset(
        dataset_dir,
        "train"
    )

    validation_data = load_dataset(
        dataset_dir,
        "validation"
    )

    print("Training samples:", len(train_data["observations"]))

    print(
        "Validation samples:",
        len(validation_data["observations"])
    )

    print("Device:", device)
    print("Seed:", seed)

    # 2. Build data loaders.

    batch_size = config["training"]["batch_size"]

    train_loader = build_data_loader(
        train_data,
        batch_size,
        shuffle=True,
        seed=seed
    )

    validation_loader = build_data_loader(
        validation_data,
        batch_size,
        shuffle=False,
        seed=seed
    )

    # 3. Build model.

    model = build_model(
        config["model"],
        device
    )

    print("\nMODEL")
    print(model)

    print(
        "Trainable parameters:",
        count_parameters(model)
    )

    # 4. Create independent training run.

    run_dir, metadata = create_run(
        experiment_dir,
        config,
        dataset_manifest
    )

    metadata.update({
        "device": str(device),

        "random_seed": seed,

        "parameter_count":
            count_parameters(model),

        "torch_version":
            str(torch.__version__),

        "training_samples":
            len(train_data["observations"]),

        "validation_samples":
            len(validation_data["observations"])
    })

    save_run_metadata(run_dir, metadata)

    print("\nRun:", run_dir.name)

    started = time.monotonic()

    # 5. Train.

    try:

        results = train_model(
            model,
            train_loader,
            validation_loader,
            config,
            device,
            run_dir
        )

        metadata.update({
            "status": "completed",

            "best_epoch":
                results["best_epoch"],

            "best_validation_loss":
                results["best_validation_loss"]
        })

    except KeyboardInterrupt:

        metadata["status"] = "interrupted"
        raise

    except Exception as error:

        metadata["status"] = "failed"
        metadata["error"] = str(error)

        raise

    finally:

        metadata["training_duration_sec"] = (
            time.monotonic() - started
        )

        metadata["ended_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        save_run_metadata(
            run_dir,
            metadata
        )

    print("\nTRAINING COMPLETED")

    print("Best epoch:", results["best_epoch"])

    print(
        "Best validation loss:",
        results["best_validation_loss"]
    )

    print("Run directory:", run_dir)


if __name__ == "__main__":
    main()