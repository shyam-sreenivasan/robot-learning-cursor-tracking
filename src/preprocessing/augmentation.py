import math
import numpy as np

from src.preprocessing.observation_transform import (
    transform_observations
)


def generate_synthetic_clicks(
    config,
    width,
    height,
    representation,
    normalization_scale=None
):

    num_samples = config["num_samples"]

    min_radius, max_radius = config[
        "target_radius_range"
    ]

    if num_samples <= 0:
        raise ValueError("Invalid sample count.")

    if not 0 < min_radius <= max_radius:
        raise ValueError("Invalid target radius range.")

    if 2 * round(max_radius) >= min(width, height):
        raise ValueError("Target does not fit workspace.")

    rng = np.random.default_rng(
        config["random_seed"]
    )

    observations = np.zeros(
        (num_samples, 5),
        dtype=np.float32
    )

    actions = np.zeros(
        (num_samples, 3),
        dtype=np.float32
    )

    for i in range(num_samples):

        # 1. Randomize target radius.

        radius = max(
            1,
            round(rng.uniform(min_radius, max_radius))
        )

        # 2. Randomize target position.

        target_x = rng.integers(
            radius, width - radius
        )

        target_y = rng.integers(
            radius, height - radius
        )

        # 3. Sample uniformly inside target disc.

        angle = rng.uniform(0, 2 * math.pi)

        distance = radius * math.sqrt(
            rng.uniform(0, 1)
        )

        cursor_x = (
            target_x + distance * math.cos(angle)
        )

        cursor_y = (
            target_y + distance * math.sin(angle)
        )

        # 4. Construct observation.

        observations[i] = [
            cursor_x,
            cursor_y,
            target_x,
            target_y,
            radius
        ]

        # 5. Demonstrate stationary clicking.

        actions[i] = [0.0, 0.0, 1.0]

    # 6. Validate generated samples.

    distances = np.linalg.norm(
        observations[:, 2:4] - observations[:, :2],
        axis=1
    )

    if np.any(distances > observations[:, 4]):
        raise ValueError(
            "Synthetic cursor outside target."
        )

    # 7. Apply existing observation transformation.

    observations = transform_observations(
        observations,
        representation,
        normalization_scale
    )

    return {
        "observations": observations,
        "actions": actions
    }