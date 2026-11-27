import csv
import json
import math
import random
import uuid

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import yaml

from src.core.environments.environment import Environment
from src.core.environments.action import Action, ButtonEvent
from src.core.tasks.click_target_task import ClickTargetTask
from src.data_acquisition.collection.episode import Episode

from src.policies.behavior_cloning import predict_action


# ============================================================
# CONFIGURATION AND SIMULATED CLOCK
# ============================================================

class EvaluationConfig:

    def __init__(self, config):
        self.config = config

    def get(self, key, default=None):

        value = self.config

        for part in key.split("."):

            if not isinstance(value, dict):
                return default

            if part not in value:
                return default

            value = value[part]

        return value


class SimulatedClock:

    def __init__(self):
        self.time = 0.0

    def __call__(self):
        return self.time

    def set_time(self, timestamp):
        self.time = timestamp


# ============================================================
# EVALUATION SCENARIOS
# ============================================================

def generate_evaluation_seeds(
    num_episodes,
    random_seed,
    excluded_seeds=None
):

    if num_episodes < 1:
        raise ValueError("Invalid episode count.")

    rng = random.Random(random_seed)

    used = set(excluded_seeds or [])
    seeds = []

    while len(seeds) < num_episodes:

        seed = rng.randrange(2**32)

        if seed not in used:
            seeds.append(seed)
            used.add(seed)

    return seeds


# ============================================================
# OBSERVATION REPRESENTATION
# ============================================================

def build_observation(state, dataset_manifest):

    observation = np.array([
        state.cursor_pos[0],
        state.cursor_pos[1],
        state.target_pos[0],
        state.target_pos[1],
        state.target_radius
    ], dtype=np.float32)

    normalization = dataset_manifest[
        "observation_normalization"
    ]

    if normalization["enabled"]:

        scale = np.array(
            normalization["scale"],
            dtype=np.float32
        )

        observation = observation / scale

    return observation


# ============================================================
# ACTION EXECUTION
# ============================================================

def execute_prediction(episode, prediction):

    state = episode.environment.get_state()

    if prediction["click"]:

        if state.button_down:

            episode.step(
                Action(button_event=ButtonEvent.RELEASE)
            )

        if not episode.is_running():
            return None

        button_event = ButtonEvent.PRESS

    elif state.button_down:

        button_event = ButtonEvent.RELEASE

    else:

        button_event = ButtonEvent.NONE

    action = Action(
        delta_x=prediction["delta_x"],
        delta_y=prediction["delta_y"],
        button_event=button_event
    )

    return episode.step(action)


# ============================================================
# SINGLE EPISODE EVALUATION
# ============================================================

def evaluate_episode(
    model,
    environment_config,
    dataset_manifest,
    inference_config,
    evaluation_config,
    seed
):

    clock = SimulatedClock()

    environment = Environment(environment_config)
    task = ClickTargetTask(environment_config)

    episode = Episode(
        environment,
        task,
        environment_config,
        clock=clock
    )

    initial_state = episode.initialize(seed)

    decision_hz = inference_config["decision_hz"]
    threshold = inference_config["click_threshold"]

    timeout = evaluation_config["timeout_sec"]

    if decision_hz <= 0 or timeout <= 0:
        raise ValueError("Invalid execution timing.")

    trajectory = []

    path_length = 0.0
    click_attempts = 0
    successful_click_error = None

    initial_distance = math.dist(
        initial_state.cursor_pos,
        initial_state.target_pos
    )

    min_distance = initial_distance

    episode.start()

    step = 0

    while episode.is_running():

        timestamp = min(
            step / decision_hz,
            timeout
        )

        clock.set_time(timestamp)

        if episode.check_timeout():
            break

        state = environment.get_state()

        observation = build_observation(
            state,
            dataset_manifest
        )

        prediction = predict_action(
            model,
            observation,
            click_threshold=threshold
        )

        if not all(
            math.isfinite(prediction[key])
            for key in [
                "delta_x",
                "delta_y",
                "click_probability"
            ]
        ):
            raise ValueError("Non-finite policy prediction.")

        interaction = execute_prediction(
            episode,
            prediction
        )

        if interaction is None:
            break

        current = interaction.current_state
        result = interaction.task_result

        displacement = math.dist(
            state.cursor_pos,
            current.cursor_pos
        )

        path_length += displacement

        distance = math.dist(
            current.cursor_pos,
            current.target_pos
        )

        min_distance = min(min_distance, distance)

        if result.attempted:
            click_attempts += 1

        if result.success:

            successful_click_error = (
                result.metrics["click_error_px"]
            )

        trajectory.append({

            "step": step,
            "timestamp_sec": timestamp,

            "previous_cursor_x": state.cursor_pos[0],
            "previous_cursor_y": state.cursor_pos[1],

            "cursor_x": current.cursor_pos[0],
            "cursor_y": current.cursor_pos[1],

            "target_x": current.target_pos[0],
            "target_y": current.target_pos[1],
            "target_radius": current.target_radius,

            "predicted_delta_x": prediction["delta_x"],
            "predicted_delta_y": prediction["delta_y"],

            "click_probability":
                prediction["click_probability"],

            "predicted_click": prediction["click"],

            "button_down": current.button_down,

            "click_attempted": result.attempted,
            "success": result.success,

            "distance_to_target_px": distance,

            "actual_displacement_px": displacement
        })

        step += 1

    episode.finalize()

    final_state = environment.get_state()

    summary = {
        **episode.get_metadata(),

        "initial_cursor": list(
            initial_state.cursor_pos
        ),

        "final_cursor": list(
            final_state.cursor_pos
        ),

        "target_position": list(
            initial_state.target_pos
        ),

        "target_radius":
            initial_state.target_radius,

        "initial_distance_px": initial_distance,

        "final_distance_px": math.dist(
            final_state.cursor_pos,
            final_state.target_pos
        ),

        "minimum_distance_px": min_distance,

        "trajectory_length_px": path_length,

        "click_attempts": click_attempts,

        "successful_click_error_px":
            successful_click_error,

        "decision_count": len(trajectory)
    }

    return summary, trajectory


# ============================================================
# AGGREGATE EVALUATION METRICS
# ============================================================

def compute_evaluation_metrics(episodes):

    def mean(values):

        values = list(values)

        return (
            float(np.mean(values))
            if values else None
        )

    successful = [
        e for e in episodes
        if e["status"] == "success"
    ]

    failed = [
        e for e in episodes
        if e["status"] == "failed"
    ]

    total = len(episodes)

    return {
        "total_episodes": total,

        "successful": len(successful),
        "failed": len(failed),

        "success_rate": (
            len(successful) / total
            if total else None
        ),

        "average_completion_sec": mean(
            e["duration_sec"]
            for e in successful
        ),

        "average_trajectory_length_px": mean(
            e["trajectory_length_px"]
            for e in episodes
        ),

        "average_initial_distance_px": mean(
            e["initial_distance_px"]
            for e in episodes
        ),

        "average_final_distance_px": mean(
            e["final_distance_px"]
            for e in episodes
        ),

        "average_minimum_distance_px": mean(
            e["minimum_distance_px"]
            for e in episodes
        ),

        "average_successful_click_error_px": mean(
            e["successful_click_error_px"]
            for e in successful
        ),

        "total_click_attempts": sum(
            e["click_attempts"]
            for e in episodes
        ),

        "episodes_with_clicks": sum(
            e["click_attempts"] > 0
            for e in episodes
        ),

        "episodes_reaching_target": sum(
            e["minimum_distance_px"]
            <= e["target_radius"]
            for e in episodes
        ),

        "total_policy_decisions": sum(
            e["decision_count"]
            for e in episodes
        )
    }


# ============================================================
# RESULT STORAGE
# ============================================================

def create_evaluation_directory(run_dir):

    evaluation_id = (
        f"evaluation_{uuid.uuid4().hex[:12]}"
    )

    path = (
        Path(run_dir)
        / "evaluations"
        / evaluation_id
    )

    path.mkdir(
        parents=True,
        exist_ok=False
    )

    (path / "episodes").mkdir()

    return path


def save_episode_trajectory(path, trajectory):

    if not trajectory:
        raise ValueError("Empty trajectory.")

    with Path(path).open("w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(trajectory[0].keys())
        )

        writer.writeheader()
        writer.writerows(trajectory)


def save_evaluation_results(
    evaluation_dir,
    manifest,
    episodes,
    metrics
):

    evaluation_dir = Path(evaluation_dir)

    (evaluation_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            manifest,
            sort_keys=False
        )
    )

    for name, data in [
        ("episodes.json", episodes),
        ("metrics.json", metrics)
    ]:

        (evaluation_dir / name).write_text(
            json.dumps(
                data,
                indent=2,
                allow_nan=False
            )
        )


# ============================================================
# COMPLETE EVALUATION RUN
# ============================================================

def evaluate_model(
    model,
    checkpoint,
    checkpoint_name,
    environment_config,
    dataset_manifest,
    experiment_config,
    run_dir,
    seeds
):

    evaluation_dir = create_evaluation_directory(
        run_dir
    )

    manifest = {
        "schema_version": "0.1",

        "evaluation_id": evaluation_dir.name,

        "run_id": Path(run_dir).name,

        "dataset_id":
            dataset_manifest["dataset"]["id"],

        "experiment_id":
            experiment_config["experiment"]["id"],

        "checkpoint": checkpoint_name,
        "checkpoint_epoch": checkpoint["epoch"],

        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "inference": checkpoint["inference_config"],

        "evaluation": experiment_config["evaluation"],

        "environment_config":
            environment_config.config,

        "scenario_seeds": seeds
    }

    episodes = []

    for index, seed in enumerate(seeds, start=1):

        summary, trajectory = evaluate_episode(
            model,
            environment_config,
            dataset_manifest,
            checkpoint["inference_config"],
            experiment_config["evaluation"],
            seed
        )

        filename = f"episode_{index:04d}.csv"

        save_episode_trajectory(
            evaluation_dir / "episodes" / filename,
            trajectory
        )

        summary["trajectory_file"] = (
            f"episodes/{filename}"
        )

        summary["evaluation_episode_number"] = index

        episodes.append(summary)

        print(
            f"Episode {index:03d} | "
            f"{summary['status']:7} | "
            f"duration={summary['duration_sec']:.2f}s | "
            f"distance={summary['final_distance_px']:.1f}px | "
            f"clicks={summary['click_attempts']}"
        )

    metrics = compute_evaluation_metrics(episodes)

    save_evaluation_results(
        evaluation_dir,
        manifest,
        episodes,
        metrics
    )

    return evaluation_dir, metrics