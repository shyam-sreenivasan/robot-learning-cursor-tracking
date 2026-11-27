import argparse

from pathlib import Path
from copy import deepcopy

import yaml

from src.data_factory.loaders.data_catalog import DataCatalog

from src.policies.behavior_cloning import (
    load_checkpoint
)

from src.evaluation.evaluation import (
    EvaluationConfig
)

from src.visualization.visualization import (
    run_live_policy
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
        "--run",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        default="best.pt"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=None
    )

    args = parser.parse_args()

    run_dir = (
        Path("data/processed")
        / args.dataset
        / "experiments"
        / args.experiment
        / "runs"
        / args.run
    )

    # 1. Load run configuration.

    experiment_config = yaml.safe_load(
        (run_dir / "config.yaml").read_text()
    )

    if args.timeout is not None:

        if args.timeout <= 0:
            raise ValueError("Timeout must be positive.")

        experiment_config["evaluation"]["timeout_sec"] = (
            args.timeout
        )

    dataset_manifest = yaml.safe_load(
        (run_dir / "dataset_manifest.yaml").read_text()
    )

    # 2. Recover original environment.

    catalog = DataCatalog("data/raw")

    source_episode = dataset_manifest[
        "episodes"
    ]["train"][0]

    session = catalog.get_session(
        source_episode["session_id"]
    )

    environment_settings = deepcopy({
        "environment": session["environment"],
        "collection": session["collection"],
        "operator": session.get("operator", {})
    })

    environment_settings["collection"][
        "episode_timeout_sec"
    ] = experiment_config["evaluation"]["timeout_sec"]

    environment_config = EvaluationConfig(
        environment_settings
    )

    # 3. Restore trained model.

    model, checkpoint = load_checkpoint(
        run_dir / "checkpoints" / args.checkpoint,
        device="cpu"
    )

    print("Dataset:", args.dataset)
    print("Experiment:", args.experiment)
    print("Run:", args.run)
    print("Checkpoint:", args.checkpoint)
    print("Epoch:", checkpoint["epoch"])

    # 4. Execute live policy.

    run_live_policy(
        model=model,
        environment_config=environment_config,
        dataset_manifest=dataset_manifest,
        inference_config=checkpoint["inference_config"],
        checkpoint_name=args.checkpoint,
        seed=args.seed
    )


if __name__ == "__main__":
    main()