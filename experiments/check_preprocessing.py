import yaml
from collections import Counter

from src.data_factory.loaders.data_catalog import DataCatalog
from src.data_factory.loaders.episode_loader import EpisodeLoader

from src.preprocessing.preprocessing import (
    select_episodes,
    preprocess_episode,
    check_reconstruction,
    compute_sample_statistics,
    split_episodes
)


with open("configs/datasets/dataset_001.yaml") as file:
    config = yaml.safe_load(file)


catalog = DataCatalog("data/raw")

episodes = select_episodes(
    catalog,
    config["source"]
)

totals = Counter()

for episode in episodes:

    loader = EpisodeLoader(
        catalog,
        episode["session_id"],
        episode["episode_id"]
    )

    manifest = catalog.get_session(
        episode["session_id"]
    )

    environment = manifest["environment"]

    processed = preprocess_episode(loader)

    stats = compute_sample_statistics(processed)

    reconstruction = check_reconstruction(
        processed,
        environment["width"],
        environment["height"]
    )

    print(
        f"{episode['episode_id'][:8]} | "
        f"samples={stats['total_samples']:4} | "
        f"clicks={stats['click_samples']:2} | "
        f"frequency={stats['effective_sampling_hz']:.2f} | "
        f"mismatches={reconstruction['mismatched_transitions']} | "
        f"final_error={reconstruction['final_position_error_px']:.4f}"
    )

    for key in [
        "total_samples",
        "movement_samples",
        "stationary_samples",
        "click_samples",
        "unassigned_actions"
    ]:
        totals[key] += stats[key]

    totals["mismatched_transitions"] += (
        reconstruction["mismatched_transitions"]
    )


print("\nDATASET STATISTICS")

print("Episodes:", len(episodes))

for key, value in totals.items():
    print(key, value)

if totals["total_samples"]:

    print(
        "Click ratio:",
        totals["click_samples"] / totals["total_samples"]
    )

    print(
        "Transition mismatch ratio:",
        totals["mismatched_transitions"]
        / totals["total_samples"]
    )


split = split_episodes(
    episodes,
    config["split"]["train_ratio"],
    config["split"]["random_seed"]
)

print("\nDATASET SPLIT")
print("Train episodes:", len(split["train"]))
print("Validation episodes:", len(split["validation"]))