import json
import shutil
import tempfile

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import yaml

from src.data_factory.loaders.episode_loader import EpisodeLoader

from src.preprocessing.observation_transform import (
    transform_observations,
    get_observation_columns,
    get_normalization_scale
)


OBSERVATION_COLUMNS = [
    "cursor_x",
    "cursor_y",
    "target_x",
    "target_y",
    "target_radius"
]

ACTION_COLUMNS = [
    "delta_x",
    "delta_y",
    "click"
]


# ============================================================
# EPISODE SELECTION
# ============================================================

def select_episodes(catalog, source_config):

    sessions = catalog.list_sessions(
        experiment_id=source_config["experiment_id"],
        config_hash=source_config["config_hash"]
    )

    outcomes = set(source_config["outcomes"])

    return [
        {
            **episode,
            "session_id": session["session_id"]
        }
        for session in sessions
        for episode in catalog.list_episodes(
            session["session_id"]
        )
        if episode["status"] in outcomes
    ]


# ============================================================
# TEMPORAL ALIGNMENT AND ACTION AGGREGATION
# ============================================================

def preprocess_episode(loader):

    states = loader.load_states()
    actions = loader.load_actions()

    if len(states) < 2:
        raise ValueError(
            "At least two states are required."
        )

    timestamps = states["timestamp"].to_numpy(
        dtype=np.float64
    )

    action_timestamps = actions["timestamp"].to_numpy(
        dtype=np.float64
    )

    displacements = actions[
        ["delta_x", "delta_y"]
    ].to_numpy(dtype=np.float64)

    button_events = actions[
        "button_event"
    ].to_numpy()

    observations = states[
        OBSERVATION_COLUMNS
    ].to_numpy(dtype=np.float32)

    intervals = np.diff(timestamps)

    if np.any(intervals <= 0):
        raise ValueError(
            "State timestamps must be strictly increasing."
        )

    boundaries = np.searchsorted(
        action_timestamps,
        timestamps,
        side="right"
    )

    targets = np.zeros(
        (len(states) - 1, 3),
        dtype=np.float32
    )

    event_counts = np.zeros(
        len(targets),
        dtype=np.int32
    )

    for i in range(len(targets)):

        start = boundaries[i]
        end = boundaries[i + 1]

        targets[i, :2] = displacements[
            start:end
        ].sum(axis=0)

        targets[i, 2] = int(
            np.any(
                button_events[start:end] == "press"
            )
        )

        event_counts[i] = end - start

    return {
        "session_id": loader.session_id,
        "episode_id": loader.episode_id,

        "metadata": loader.get_metadata(),

        "observations": observations[:-1],
        "actions": targets,

        "next_observations": observations[1:],

        "timestamps_sec":
            timestamps[:-1] - timestamps[0],

        "intervals_sec": intervals,

        "frame_ids": states[
            "frame_id"
        ].to_numpy()[:-1],

        "event_counts": event_counts,

        "raw_action_count": len(actions),

        "unassigned_actions": int(
            boundaries[0]
            + len(actions)
            - boundaries[-1]
        )
    }


# ============================================================
# OBSERVATION NORMALIZATION
# ============================================================

def normalize_observations(
    observations,
    width,
    height
):

    scale = get_normalization_scale(
        "absolute_state",
        width,
        height
    )

    return transform_observations(
        observations,
        "absolute_state",
        scale
    )


# ============================================================
# AGGREGATION RECONSTRUCTION CHECK
# ============================================================

def check_reconstruction(
    processed,
    width,
    height,
    tolerance_px=0.01
):

    observations = processed["observations"]
    actions = processed["actions"]
    next_observations = processed["next_observations"]

    actual = next_observations[:, :2]

    lower = np.array([0, 0])

    upper = np.array([
        width - 1,
        height - 1
    ])

    # Check each transition independently.

    predicted = np.clip(
        observations[:, :2] + actions[:, :2],
        lower,
        upper
    )

    errors = np.linalg.norm(
        predicted - actual,
        axis=1
    )

    # Reconstruct the full trajectory sequentially.

    cursor = observations[0, :2].copy()

    reconstructed = [cursor.copy()]

    for action in actions:

        cursor = np.clip(
            cursor + action[:2],
            lower,
            upper
        )

        reconstructed.append(cursor.copy())

    reconstructed = np.array(reconstructed)

    trajectory_errors = np.linalg.norm(
        reconstructed[1:] - actual,
        axis=1
    )

    return {
        "total_transitions": len(errors),

        "matching_transitions": int(
            np.sum(errors <= tolerance_px)
        ),

        "mismatched_transitions": int(
            np.sum(errors > tolerance_px)
        ),

        "mean_transition_error_px":
            float(errors.mean()),

        "max_transition_error_px":
            float(errors.max()),

        "mean_trajectory_error_px":
            float(trajectory_errors.mean()),

        "final_position_error_px":
            float(trajectory_errors[-1]),

        "reconstructed_trajectory":
            reconstructed
    }


# ============================================================
# PROCESSED EPISODE STATISTICS
# ============================================================

def compute_sample_statistics(processed):

    actions = processed["actions"]
    intervals = processed["intervals_sec"]

    movement = np.linalg.norm(
        actions[:, :2],
        axis=1
    )

    return {
        "total_samples": len(actions),

        "movement_samples": int(
            np.sum(movement > 0)
        ),

        "stationary_samples": int(
            np.sum(movement == 0)
        ),

        "click_samples": int(
            np.sum(actions[:, 2] == 1)
        ),

        "click_ratio": float(
            actions[:, 2].mean()
        ),

        "mean_displacement_px":
            float(movement.mean()),

        "max_displacement_px":
            float(movement.max()),

        "effective_sampling_hz":
            float(1 / intervals.mean()),

        "mean_interval_ms":
            float(intervals.mean() * 1000),

        "max_interval_ms":
            float(intervals.max() * 1000),

        "unassigned_actions":
            processed["unassigned_actions"]
    }


# ============================================================
# EPISODE-LEVEL DATASET SPLIT
# ============================================================

def split_episodes(
    episodes,
    train_ratio=0.8,
    seed=42
):

    if len(episodes) < 2:
        raise ValueError(
            "At least two episodes are required."
        )

    if not 0 < train_ratio < 1:
        raise ValueError("Invalid training ratio.")

    rng = np.random.default_rng(seed)

    indices = rng.permutation(len(episodes))

    split_index = max(
        1,
        min(
            len(episodes) - 1,
            int(len(episodes) * train_ratio)
        )
    )

    return {
        "train": [
            episodes[i]
            for i in indices[:split_index]
        ],

        "validation": [
            episodes[i]
            for i in indices[split_index:]
        ]
    }


# ============================================================
# BUILD PROCESSED SPLIT
# ============================================================

def build_split(catalog, episodes, config):

    buffers = {
        "observations": [],
        "actions": [],
        "timestamps_sec": [],
        "intervals_sec": [],
        "session_ids": [],
        "episode_ids": [],
        "frame_ids": []
    }

    diagnostics = []

    for episode in episodes:

        session_id = episode["session_id"]
        episode_id = episode["episode_id"]

        loader = EpisodeLoader(
            catalog,
            session_id,
            episode_id
        )

        manifest = catalog.get_session(session_id)
        environment = manifest["environment"]

        width = environment["width"]
        height = environment["height"]

        processed = preprocess_episode(loader)

        # Validate action aggregation using raw absolute
        # coordinates before applying any transformation.

        reconstruction = check_reconstruction(
            processed,
            width,
            height
        )

        if processed["unassigned_actions"]:
            raise ValueError(
                f"Unassigned actions: {episode_id}"
            )

        if reconstruction["mismatched_transitions"]:
            raise ValueError(
                f"Reconstruction mismatch: {episode_id}"
            )

        # Apply configured observation representation.

        representation = config[
            "observation_representation"
        ]

        scale = None

        if config["normalize_observations"]:

            scale = get_normalization_scale(
                representation,
                width,
                height
            )

        observations = transform_observations(
            processed["observations"],
            representation,
            scale
        )

        count = len(observations)

        buffers["observations"].append(observations)
        buffers["actions"].append(processed["actions"])

        for key in [
            "timestamps_sec",
            "intervals_sec",
            "frame_ids"
        ]:
            buffers[key].append(processed[key])

        buffers["session_ids"].append(
            np.full(count, session_id)
        )

        buffers["episode_ids"].append(
            np.full(count, episode_id)
        )

        diagnostics.append({
            "session_id": session_id,
            "episode_id": episode_id,
            "status": episode["status"],

            **compute_sample_statistics(processed),

            "mismatched_transitions":
                reconstruction["mismatched_transitions"],

            "final_position_error_px":
                reconstruction["final_position_error_px"]
        })

    arrays = {
        key: np.concatenate(values, axis=0)
        for key, values in buffers.items()
    }

    return arrays, diagnostics


# ============================================================
# DATASET STATISTICS
# ============================================================

def append_synthetic_samples(arrays, synthetic):
    """Append independent supervised rows, not recorded episodes."""
    n = len(arrays["actions"])
    m = len(synthetic["actions"])
    if m < 1 or synthetic["actions"].shape != (m, 3):
        raise ValueError("Invalid synthetic action shape.")
    if synthetic["observations"].shape != (m, arrays["observations"].shape[1]):
        raise ValueError("Invalid synthetic observation shape.")
    if not np.isfinite(synthetic["observations"]).all():
        raise ValueError("Non-finite synthetic observations.")
    if not np.all(synthetic["actions"] == [0, 0, 1]):
        raise ValueError("Expected stationary click actions.")
    if "sample_source" in arrays:
        raise ValueError("Expected unaugmented human samples.")
    additions = {
        "observations": synthetic["observations"],
        "actions": synthetic["actions"],
        "timestamps_sec": np.full(m, np.nan),
        "intervals_sec": np.full(m, np.nan),
        "session_ids": np.full(m, ""),
        "episode_ids": np.full(m, ""),
        "frame_ids": np.full(m, -1, dtype=np.int64),
    }
    if set(arrays) != set(additions):
        raise ValueError("Unexpected dataset array schema.")
    result = {key: np.concatenate([value, additions[key].astype(value.dtype)])
              for key, value in arrays.items()}
    result["sample_source"] = np.concatenate([
        np.full(n, "human", dtype="U15"),
        np.full(m, "synthetic_click", dtype="U15")
    ])
    return result


def compute_dataset_statistics(arrays, diagnostics):

    actions = arrays["actions"]

    displacement = np.linalg.norm(
        actions[:, :2],
        axis=1
    )

    sources = arrays.get("sample_source", np.full(len(actions), "human"))
    human = sources == "human"
    intervals = arrays["intervals_sec"][human]
    if not np.all(np.isfinite(intervals) & (intervals > 0)):
        raise ValueError("Invalid recorded sample intervals.")

    return {
        "human_samples": int(human.sum()),
        "synthetic_samples": int((~human).sum()),
        "human_click_samples": int(np.sum(human & (actions[:, 2] == 1))),
        "synthetic_click_samples": int(np.sum(~human & (actions[:, 2] == 1))),
        "total_episodes": len(diagnostics),
        "total_samples": len(actions),

        "observation_shape":
            list(arrays["observations"].shape),

        "action_shape":
            list(actions.shape),

        "movement_samples":
            int(np.sum(displacement > 0)),

        "stationary_samples":
            int(np.sum(displacement == 0)),

        "click_samples":
            int(np.sum(actions[:, 2] == 1)),

        "click_ratio":
            float(actions[:, 2].mean()),

        "mean_displacement_px":
            float(displacement.mean()),

        "max_displacement_px":
            float(displacement.max()),

        "effective_sampling_hz":
            float(1 / intervals.mean()) if len(intervals) else None,

        "successful_episodes": sum(
            e["status"] == "success"
            for e in diagnostics
        ),

        "failed_episodes": sum(
            e["status"] == "failed"
            for e in diagnostics
        ),

        "aborted_episodes": sum(
            e["status"] == "aborted"
            for e in diagnostics
        ),

        "unassigned_actions": sum(
            e["unassigned_actions"]
            for e in diagnostics
        ),

        "mismatched_transitions": sum(
            e["mismatched_transitions"]
            for e in diagnostics
        )
    }


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    output_root,
    config,
    splits,
    arrays,
    statistics
):

    output_root = Path(output_root)

    dataset_id = config["dataset"]["id"]

    if (
        not dataset_id
        or Path(dataset_id).name != dataset_id
        or dataset_id in {".", ".."}
    ):
        raise ValueError("Invalid dataset ID.")

    output_dir = output_root / dataset_id

    if output_dir.exists():
        raise FileExistsError(
            f"Dataset already exists: {output_dir}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    representation = config["preprocessing"][
        "observation_representation"
    ]

    manifest = {
        "schema_version": "0.1",

        "dataset": config["dataset"],

        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "source": config["source"],

        "preprocessing": config["preprocessing"],

        "split": config["split"],

        "observation_columns":
            get_observation_columns(representation),

        "action_columns": ACTION_COLUMNS,

        "observation_normalization": {
            "enabled": config["preprocessing"][
                "normalize_observations"
            ],
            "scale": config["normalization_scale"]
        },

        "action_units": [
            "pixels",
            "pixels",
            "binary"
        ],

        "episodes": {
            name: [
                {
                    "session_id": e["session_id"],
                    "episode_id": e["episode_id"],
                    "status": e["status"]
                }
                for e in episodes
            ]
            for name, episodes in splits.items()
        }
    }

    if config.get("augmentation", {}).get("enabled", False):
        manifest["augmentation"] = config["augmentation"]
        manifest["sample_metadata"] = {
            "sample_source": ["human", "synthetic_click"],
            "synthetic_missing_metadata": {
                "session_ids": "", "episode_ids": "",
                "frame_ids": -1, "timestamps_sec": "NaN",
                "intervals_sec": "NaN"
            }
        }

    # Write to a temporary directory first.

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{dataset_id}_",
            dir=output_root
        )
    )

    try:

        for name, data in arrays.items():

            np.savez_compressed(
                temp_dir / f"{name}.npz",
                **data
            )

        (temp_dir / "manifest.yaml").write_text(
            yaml.safe_dump(
                manifest,
                sort_keys=False
            )
        )

        (temp_dir / "statistics.json").write_text(
            json.dumps(
                statistics,
                indent=2,
                allow_nan=False
            )
        )

        (temp_dir / "experiments").mkdir()

        temp_dir.rename(output_dir)

    except Exception:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        raise

    return output_dir


# ============================================================
# LOAD PROCESSED DATASET
# ============================================================

def load_dataset(dataset_dir, split="train"):

    if split not in {"train", "validation"}:
        raise ValueError("Unknown dataset split.")

    path = Path(dataset_dir) / f"{split}.npz"

    with np.load(path, allow_pickle=False) as data:

        return {
            key: data[key]
            for key in data.files
        }


def load_dataset_manifest(dataset_dir):

    path = Path(dataset_dir) / "manifest.yaml"

    return yaml.safe_load(path.read_text())


def load_dataset_statistics(dataset_dir):

    path = Path(dataset_dir) / "statistics.json"

    return json.loads(path.read_text())