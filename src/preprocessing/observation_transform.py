import numpy as np


def get_observation_columns(representation):

    columns = [
        "cursor_x",
        "cursor_y",
        "target_x",
        "target_y",
        "target_radius"
    ]

    if representation == "absolute_state":
        return columns

    if representation == "augmented_spatial":
        return columns + [
            "relative_x",
            "relative_y"
        ]

    raise ValueError(
        f"Unsupported representation: {representation}"
    )


def get_normalization_scale(
    representation,
    width,
    height
):

    scale = [
        width,
        height,
        width,
        height,
        min(width, height)
    ]

    if representation == "augmented_spatial":
        scale.extend([width, height])

    elif representation != "absolute_state":
        raise ValueError(
            f"Unsupported representation: {representation}"
        )

    return scale


def transform_observations(
    observations,
    representation,
    normalization_scale=None
):

    observations = np.asarray(
        observations,
        dtype=np.float32
    )

    if observations.shape[-1] != 5:
        raise ValueError(
            "Expected five absolute state features."
        )

    if representation == "absolute_state":

        transformed = observations.copy()

    elif representation == "augmented_spatial":

        relative_x = (
            observations[..., 2:3]
            - observations[..., 0:1]
        )

        relative_y = (
            observations[..., 3:4]
            - observations[..., 1:2]
        )

        transformed = np.concatenate(
            [
                observations,
                relative_x,
                relative_y
            ],
            axis=-1
        )

    else:
        raise ValueError(
            f"Unsupported representation: {representation}"
        )

    if normalization_scale is not None:

        transformed = transformed / np.asarray(
            normalization_scale,
            dtype=np.float32
        )

    return transformed