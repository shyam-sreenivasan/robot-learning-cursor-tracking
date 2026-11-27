import pytest

from src.config.config_manager import ConfigManager
from src.environments.environment import Environment
from src.environments.action import Action, ButtonEvent
from src.tasks.click_target_task import ClickTargetTask
from src.collection.episode import Episode
from src.collection.episode_status import EpisodeStatus

class FakeClock:

    def __init__(self):
        self.time = 0.0

    def __call__(self):
        return self.time

    def advance(self, seconds):
        self.time += seconds

@pytest.fixture
def setup_episode():
    config = ConfigManager("configs/experiment_001.yaml")
    clock = FakeClock()

    env = Environment(config)
    task = ClickTargetTask(config)

    episode = Episode(env, task, config, clock=clock)
    episode.initialize(seed=42)

    return episode, clock

def test_initialization(setup_episode):
    episode, clock = setup_episode

    assert episode.get_status() == EpisodeStatus.READY


def test_start(setup_episode):
    episode, clock = setup_episode

    episode.start()

    assert episode.get_status() == EpisodeStatus.RUNNING


def test_timeout(setup_episode):
    episode, clock = setup_episode
    episode.start()

    clock.advance(episode.timeout)

    assert episode.check_timeout()
    assert episode.get_status() == EpisodeStatus.FAILED
    assert episode.termination_reason == "timeout"


def test_duration(setup_episode):
    episode, clock = setup_episode
    episode.start()

    clock.advance(3)

    assert episode.get_duration() == 3


def test_abort(setup_episode):
    episode, clock = setup_episode
    episode.start()

    clock.advance(2)
    episode.abort()

    assert episode.get_status() == EpisodeStatus.ABORTED
    assert episode.get_duration() == 2

    clock.advance(5)

    assert episode.get_duration() == 2


def test_failed_click(setup_episode):
    episode, clock = setup_episode
    episode.start()

    result = episode.step(
        Action(button_event=ButtonEvent.PRESS)
    )

    assert result.task_result.attempted
    assert not result.task_result.success
    assert episode.get_status() == EpisodeStatus.RUNNING


def test_successful_click(setup_episode):
    episode, clock = setup_episode
    episode.start()

    state = episode.environment.get_state()

    dx = state.target_pos[0] - state.cursor_pos[0]
    dy = state.target_pos[1] - state.cursor_pos[1]

    episode.step(Action(dx, dy))

    clock.advance(2)

    result = episode.step(
        Action(button_event=ButtonEvent.PRESS)
    )

    assert result.task_result.success
    assert episode.get_status() == EpisodeStatus.SUCCESS
    assert episode.get_duration() == 2


def test_action_after_termination(setup_episode):
    episode, clock = setup_episode
    episode.start()
    episode.abort()

    with pytest.raises(RuntimeError):
        episode.step(Action(10, 10))


def test_action_after_deadline(setup_episode):
    episode, clock = setup_episode
    episode.start()

    previous_state = episode.environment.get_state()

    clock.advance(episode.timeout)

    result = episode.step(Action(10, 10))

    assert result is None
    assert episode.get_status() == EpisodeStatus.FAILED
    assert episode.environment.get_state() == previous_state


def test_start_twice(setup_episode):
    episode, clock = setup_episode
    episode.start()

    with pytest.raises(RuntimeError):
        episode.start()