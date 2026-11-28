import argparse
import yaml
import numpy as np

from pathlib import Path
from pprint import pprint

from src.data_factory.loaders.data_catalog import DataCatalog

from src.preprocessing.observation_transform import (
    get_normalization_scale
)

from src.preprocessing.preprocessing import (
    select_episodes,
    split_episodes,
    build_split,
    compute_dataset_statistics,
    save_dataset,
    load_dataset_manifest,
    load_dataset
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

    output_root = Path("data/processed")

    # 1. Select episodes.

    episodes = select_episodes(
        catalog,
        config["source"]
    )

    # 2. Establish dataset split.

    reference_dataset = config["split"].get(
        "reuse_from"
    )

    reference_dir = None

    if reference_dataset:

        reference_dir = output_root / reference_dataset

        reference_manifest = load_dataset_manifest(
            reference_dir
        )

        available = {
            (e["session_id"], e["episode_id"]): e
            for e in episodes
        }

        splits = {}

        for name, selected in reference_manifest[
            "episodes"
        ].items():

            splits[name] = []

            for episode in selected:

                key = (
                    episode["session_id"],
                    episode["episode_id"]
                )

                if key not in available:
                    raise ValueError(
                        f"Reference episode unavailable: {key}"
                    )

                current = available[key]

                if current["status"] != episode["status"]:
                    raise ValueError(
                        f"Reference episode status changed: {key}"
                    )

                splits[name].append(current)

        print("Reusing split from:", reference_dataset)

    else:

        splits = split_episodes(
            episodes,
            train_ratio=config["split"]["train_ratio"],
            seed=config["split"]["random_seed"]
        )

    episodes = [
        episode
        for selected in splits.values()
        for episode in selected
    ]

    print("Selected episodes:", len(episodes))

    # 3. Establish normalization parameters.

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

    config["normalization_scale"] = (
        get_normalization_scale(
            config["preprocessing"][
                "observation_representation"
            ],
            width,
            height
        )
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

    # 5. Verify controlled dataset equivalence.

    if reference_dir is not None:

        for name, data in arrays.items():

            reference = load_dataset(
                reference_dir,
                name
            )

            for key in [
                "actions",
                "timestamps_sec",
                "intervals_sec",
                "session_ids",
                "episode_ids",
                "frame_ids"
            ]:

                if not np.array_equal(
                    reference[key],
                    data[key]
                ):
                    raise ValueError(
                        f"Reference mismatch: {name}/{key}"
                    )

            print(
                f"{name}: reference data verified."
            )

    # 6. Compute overall statistics.

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

    # 7. Persist dataset.

    output_dir = save_dataset(
        output_root,
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