import argparse
from pathlib import Path
from copy import deepcopy
from pprint import pprint

import yaml

from src.data_factory.loaders.data_catalog import DataCatalog

from src.policies.behavior_cloning import (
    load_checkpoint
)

from src.evaluation.evaluation import (
    EvaluationConfig,
    generate_evaluation_seeds,
    evaluate_model
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

    # 1. Recover original collection conditions.

    catalog = DataCatalog("data/raw")

    source_episodes = [
        episode
        for episodes in dataset_manifest["episodes"].values()
        for episode in episodes
    ]

    original_session = catalog.get_session(
        source_episodes[0]["session_id"]
    )

    environment_settings = deepcopy({
        "environment": original_session["environment"],
        "collection": original_session["collection"],
        "operator": original_session.get("operator", {})
    })

    evaluation_config = experiment_config["evaluation"]

    environment_settings["collection"][
        "episode_timeout_sec"
    ] = evaluation_config["timeout_sec"]

    environment_config = EvaluationConfig(
        environment_settings
    )

    # 2. Exclude seeds from the demonstration dataset.

    excluded_seeds = set()

    for episode in source_episodes:

        registered = catalog.get_episode(
            episode["session_id"],
            episode["episode_id"]
        )

        excluded_seeds.add(registered["seed"])

    seeds = generate_evaluation_seeds(
        evaluation_config["num_episodes"],
        evaluation_config["random_seed"],
        excluded_seeds
    )

    # 3. Restore checkpoint.

    checkpoint_path = (
        run_dir
        / "checkpoints"
        / args.checkpoint
    )

    model, checkpoint = load_checkpoint(
        checkpoint_path,
        device="cpu"
    )

    print("Dataset:", args.dataset)
    print("Experiment:", args.experiment)
    print("Run:", args.run)
    print("Checkpoint:", args.checkpoint)
    print("Epoch:", checkpoint["epoch"])

    # 4. Execute evaluation.

    evaluation_dir, metrics = evaluate_model(
        model,
        checkpoint,
        args.checkpoint,
        environment_config,
        dataset_manifest,
        experiment_config,
        run_dir,
        seeds
    )

    print("\nEVALUATION RESULTS")
    pprint(metrics)

    print("\nResults saved to:")
    print(evaluation_dir)


if __name__ == "__main__":
    main()