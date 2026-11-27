import argparse
import yaml
import numpy as np

from pprint import pprint

from src.data_factory.loaders.data_catalog import DataCatalog

from src.preprocessing.preprocessing import (
    select_episodes,
    split_episodes,
    build_split,
    compute_dataset_statistics,
    save_dataset
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/datasets/dataset_001.yaml"
    )

    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    catalog = DataCatalog("data/raw")

    # 1. Select episodes.

    episodes = select_episodes(
        catalog,
        config["source"]
    )

    print("Selected episodes:", len(episodes))

    # 2. Establish normalization parameters.

    workspaces = set()

    for session_id in {
        e["session_id"] for e in episodes
    }:

        environment = catalog.get_session(
            session_id
        )["environment"]

        workspaces.add((
            environment["width"],
            environment["height"]
        ))

    if len(workspaces) != 1:
        raise ValueError(
            "Expected one consistent workspace."
        )

    width, height = workspaces.pop()

    config["normalization_scale"] = [
        width,
        height,
        width,
        height,
        min(width, height)
    ]

    # 3. Split by episode.

    splits = split_episodes(
        episodes,
        train_ratio=config["split"]["train_ratio"],
        seed=config["split"]["random_seed"]
    )

    arrays = {}
    statistics = {}
    all_diagnostics = []

    # 4. Process each split.

    for name, selected in splits.items():

        print(f"\nProcessing {name} split")

        data, diagnostics = build_split(
            catalog,
            selected,
            config["preprocessing"]
        )

        arrays[name] = data

        statistics[name] = compute_dataset_statistics(
            data,
            diagnostics
        )

        all_diagnostics.extend(diagnostics)

        pprint(statistics[name])

    # 5. Compute overall statistics.

    combined = {
        key: np.concatenate([
            arrays["train"][key],
            arrays["validation"][key]
        ])
        for key in arrays["train"]
    }

    statistics["overall"] = compute_dataset_statistics(
        combined,
        all_diagnostics
    )

    statistics["episodes"] = all_diagnostics

    # 6. Persist dataset.

    output_dir = save_dataset(
        "data/processed",
        config,
        splits,
        arrays,
        statistics
    )

    print("\nDataset saved successfully.")
    print("Location:", output_dir)

    pprint(statistics["overall"])


if __name__ == "__main__":
    main()