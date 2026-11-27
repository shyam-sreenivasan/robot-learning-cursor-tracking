import pytest
import numpy as np

from src.core.config.config_manager import ConfigManager
from src.core.environments.environment import Environment
from src.core.environments.action import Action
from src.data_acquisition.interfaces.pygame_renderer import PygameRenderer


@pytest.fixture
def setup_renderer():

    config = ConfigManager("configs/experiment_001.yaml")

    env = Environment(config)
    renderer = PygameRenderer(config)

    state = env.reset(seed=42)
    renderer.render(state)

    return config, env, renderer


def test_image_dimensions(setup_renderer):

    config, env, renderer = setup_renderer

    image = renderer.get_observation()

    assert image.shape == (
        config.get("environment.height"),
        config.get("environment.width"),
        3
    )


def test_image_dtype(setup_renderer):

    config, env, renderer = setup_renderer

    assert renderer.get_observation().dtype == np.uint8


def test_background_color(setup_renderer):

    config, env, renderer = setup_renderer

    image = renderer.get_observation()

    assert np.array_equal(
        image[0, 0],
        config.get("environment.background_color")
    )


def test_target_rendering(setup_renderer):

    config, env, renderer = setup_renderer

    x, y = env.get_state().target_pos

    image = renderer.get_observation()

    assert np.array_equal(
        image[y, x],
        config.get("environment.target.color")
    )


def test_cursor_rendering(setup_renderer):

    config, env, renderer = setup_renderer

    x, y = map(round, env.get_state().cursor_pos)

    image = renderer.get_observation()

    assert np.array_equal(
        image[y, x],
        config.get("environment.cursor.color")
    )


def test_observation_isolation(setup_renderer):

    config, env, renderer = setup_renderer

    first_image = renderer.get_observation()
    original = first_image.copy()

    state = env.step(Action(50, 20))

    renderer.render(state)

    second_image = renderer.get_observation()

    assert np.array_equal(first_image, original)
    assert not np.array_equal(first_image, second_image)