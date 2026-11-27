import math

import numpy as np

from src.data_factory.loaders.episode_loader import EpisodeLoader


# ============================================================
# GLOBAL INVENTORY
# ============================================================

def get_overview(catalog):

    experiments = catalog.list_experiments()
    sessions = catalog.list_sessions()

    episodes = [
        {
            **episode,
            "session_id": session["session_id"]
        }
        for session in sessions
        for episode in catalog.list_episodes(
            session["session_id"]
        )
    ]

    return {
        "total_experiments": len({
            e["experiment_id"] for e in experiments
        }),
        "total_configurations": len(experiments),
        "total_sessions": len(sessions),
        **compute_statistics(episodes)
    }


# ============================================================
# AGGREGATE STATISTICS
# ============================================================

def compute_statistics(episodes):

    successful = [
        e for e in episodes
        if e["status"] == "success"
    ]

    failed = [
        e for e in episodes
        if e["status"] == "failed"
    ]

    aborted = [
        e for e in episodes
        if e["status"] == "aborted"
    ]

    evaluable = len(successful) + len(failed)

    durations = [
        float(e["duration_sec"])
        for e in successful
    ]

    fastest = min(
        successful,
        key=lambda e: float(e["duration_sec"]),
        default=None
    )

    slowest = max(
        successful,
        key=lambda e: float(e["duration_sec"]),
        default=None
    )

    return {
        "total_episodes": len(episodes),
        "successful": len(successful),
        "failed": len(failed),
        "aborted": len(aborted),
        "evaluable": evaluable,

        "success_rate": (
            len(successful) / evaluable
            if evaluable else None
        ),

        "total_duration_sec": sum(
            float(e["duration_sec"])
            for e in episodes
        ),

        "average_completion_sec": (
            sum(durations) / len(durations)
            if durations else None
        ),

        "fastest_episode": fastest,
        "slowest_episode": slowest
    }


# ============================================================
# EPISODE ANALYTICS
# ============================================================

def get_episode_summary(loader):

    metadata = loader.get_metadata()
    states = loader.load_states()

    initial = states.iloc[0]
    final = states.iloc[-1]

    dx = float(
        initial["target_x"] - initial["cursor_x"]
    )

    dy = float(
        initial["target_y"] - initial["cursor_y"]
    )

    return {
        **metadata,

        "session_id": loader.session_id,

        "initial_cursor": [
            float(initial["cursor_x"]),
            float(initial["cursor_y"])
        ],

        "target_position": [
            float(initial["target_x"]),
            float(initial["target_y"])
        ],

        "target_radius": float(
            initial["target_radius"]
        ),

        "initial_distance_px": math.hypot(dx, dy),

        "initial_direction_deg": (
            math.degrees(math.atan2(dy, dx)) % 360
            if dx or dy else None
        ),

        "final_cursor": [
            float(final["cursor_x"]),
            float(final["cursor_y"])
        ],

        "first_frame_id": int(initial["frame_id"]),
        "last_frame_id": int(final["frame_id"])
    }


def get_episode_trajectory(loader):

    states = loader.load_states()

    return {
        "timestamps_sec": (
            states["timestamp"]
            - states["timestamp"].iloc[0]
        ).tolist(),

        "cursor_x": states["cursor_x"].tolist(),
        "cursor_y": states["cursor_y"].tolist(),

        "target_position": [
            float(states.iloc[0]["target_x"]),
            float(states.iloc[0]["target_y"])
        ],

        "target_radius": float(
            states.iloc[0]["target_radius"]
        )
    }


# ============================================================
# SPATIAL COVERAGE
# ============================================================

def compute_spatial_coverage(
    episodes,
    width,
    height,
    bins_x=8,
    bins_y=6
):

    if min(width, height, bins_x, bins_y) <= 0:
        raise ValueError(
            "Workspace dimensions and bin counts must be positive."
        )

    def compute_grid(positions):

        grid = np.zeros(
            (bins_y, bins_x),
            dtype=int
        )

        for x, y in positions:

            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"Position outside workspace: {(x, y)}"
                )

            col = int(x / width * bins_x)
            row = int(y / height * bins_y)

            grid[row, col] += 1

        occupied = int(np.count_nonzero(grid))
        total = bins_x * bins_y

        return {
            "positions": positions,
            "grid": grid.tolist(),
            "occupied_bins": occupied,
            "total_bins": total,
            "coverage_ratio": occupied / total
        }

    return {
        "workspace": {
            "width": width,
            "height": height
        },

        "bins_x": bins_x,
        "bins_y": bins_y,

        "cursor_coverage": compute_grid([
            e["initial_cursor"]
            for e in episodes
        ]),

        "target_coverage": compute_grid([
            e["target_position"]
            for e in episodes
        ]),

        "initial_distances_px": [
            e["initial_distance_px"]
            for e in episodes
        ],

        "initial_directions_deg": [
            e["initial_direction_deg"]
            for e in episodes
        ],

        "target_radii_px": [
            e["target_radius"]
            for e in episodes
        ]
    }


# ============================================================
# SESSION ANALYTICS
# ============================================================

def get_session_stats(
    catalog,
    session_id,
    bins_x=8,
    bins_y=6
):

    manifest = catalog.get_session(session_id)
    registered = catalog.list_episodes(session_id)

    episodes = [
        {
            **get_episode_summary(
                EpisodeLoader(
                    catalog,
                    session_id,
                    e["episode_id"]
                )
            ),
            "batch_number": e["batch_number"]
        }
        for e in registered
    ]

    environment = manifest["environment"]

    return {
        "session_id": session_id,

        "experiment_id": manifest["experiment"]["id"],

        "config_hash": manifest["execution"]["config_hash"],

        "operator_id": manifest.get(
            "operator", {}
        ).get("id"),

        "statistics": compute_statistics(episodes),

        "coverage": compute_spatial_coverage(
            episodes,
            environment["width"],
            environment["height"],
            bins_x=bins_x,
            bins_y=bins_y
        ),

        "episodes": episodes
    }


# ============================================================
# EXPERIMENT ANALYTICS
# ============================================================

def get_experiment_stats(
    catalog,
    experiment_id,
    config_hash,
    bins_x=8,
    bins_y=6
):

    sessions = catalog.list_sessions(
        experiment_id=experiment_id,
        config_hash=config_hash
    )

    if not sessions:
        raise KeyError(
            "Experiment configuration not found"
        )

    session_results = [
        get_session_stats(
            catalog,
            session["session_id"],
            bins_x=bins_x,
            bins_y=bins_y
        )
        for session in sessions
    ]

    episodes = [
        episode
        for session in session_results
        for episode in session["episodes"]
    ]

    workspace = session_results[0]["coverage"]["workspace"]

    if any(
        result["coverage"]["workspace"] != workspace
        for result in session_results
    ):
        raise ValueError(
            "Inconsistent workspace dimensions across sessions."
        )

    return {
        "experiment_id": experiment_id,
        "config_hash": config_hash,

        "experiment_name": sessions[0]["experiment_name"],
        "experiment_version": sessions[0]["experiment_version"],

        "total_sessions": len(sessions),

        "statistics": compute_statistics(episodes),

        "coverage": compute_spatial_coverage(
            episodes,
            workspace["width"],
            workspace["height"],
            bins_x=bins_x,
            bins_y=bins_y
        ),

        "sessions": [
            {
                **session,
                "statistics": result["statistics"]
            }
            for session, result in zip(
                sessions, session_results
            )
        ],

        "episodes": episodes
    }