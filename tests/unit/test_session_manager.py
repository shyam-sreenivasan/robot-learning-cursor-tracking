import pytest
import yaml

from src.config.config_manager import ConfigManager

from src.collection.session_manager import SessionManager
from src.collection.session_status import SessionStatus
from src.collection.episode_status import EpisodeStatus
from src.collection.recorder_status import RecorderStatus

from src.environments.action import Action, ButtonEvent


@pytest.fixture
def config(tmp_path):

    config = ConfigManager("configs/experiment_001.yaml")

    config.set("storage.raw_data_dir", str(tmp_path))
    config.set("collection.episodes_per_batch", 3)
    config.set("collection.random_seed", 42)

    return config


@pytest.fixture
def session(config):

    session = SessionManager(config)
    session.start()

    return session


def complete_successful_episode(session):

    episode = session.create_episode()
    episode.start()

    state = episode.environment.get_state()

    episode.step(
        Action(
            delta_x=state.target_pos[0] - state.cursor_pos[0],
            delta_y=state.target_pos[1] - state.cursor_pos[1],
            button_event=ButtonEvent.PRESS
        )
    )

    episode.finalize()

    return session.complete_episode()

def test_configuration_isolation(config):

    session = SessionManager(config)

    config.set("collection.episodes_per_batch", 20)

    assert session.config.get(
        "collection.episodes_per_batch"
    ) == 3


def test_create_episode(session):

    episode = session.create_episode()

    assert episode.get_status() == EpisodeStatus.READY

    assert (
        episode.recorder.get_status()
        == RecorderStatus.IDLE
    )

    assert session.current_episode is episode


def test_concurrent_episode_rejected(session):

    session.create_episode()

    with pytest.raises(RuntimeError):
        session.create_episode()


def test_unfinished_episode_registration(session):

    session.create_episode()

    with pytest.raises(RuntimeError):
        session.complete_episode()


def test_initialization(config):

    session = SessionManager(config)

    assert session.get_status() == SessionStatus.CREATED
    assert session.current_episode is None


def test_start_session(config):

    session = SessionManager(config)
    session.start()

    assert session.get_status() == SessionStatus.ACTIVE
    assert session.session_dir.exists()

    assert (session.session_dir / "session.yaml").exists()


def test_configuration_isolation(config):

    session = SessionManager(config)

    config.set("collection.episodes_per_batch", 20)

    assert session.config.get(
        "collection.episodes_per_batch"
    ) == 3


def test_create_episode(session):

    episode = session.create_episode()

    assert episode.get_status() == EpisodeStatus.READY

    assert (
        episode.recorder.get_status()
        == RecorderStatus.IDLE
    )

    assert session.current_episode is episode


def test_concurrent_episode_rejected(session):

    session.create_episode()

    with pytest.raises(RuntimeError):
        session.create_episode()


def test_unfinished_episode_registration(session):

    session.create_episode()

    with pytest.raises(RuntimeError):
        session.complete_episode()

def test_successful_episode(session):

    complete_successful_episode(session)

    stats = session.get_statistics()

    assert stats["total"] == 1
    assert stats["successful"] == 1
    assert stats["success_rate"] == 1.0
    assert session.current_episode is None


def test_aborted_episode(session):

    episode = session.create_episode()

    episode.start()
    episode.abort()
    episode.finalize()

    session.complete_episode()

    stats = session.get_statistics()

    assert stats["total"] == 1
    assert stats["aborted"] == 1
    assert stats["evaluable"] == 0
    assert stats["success_rate"] is None
    assert not session.is_batch_complete()


def test_batch_completion(session):

    for _ in range(3):
        complete_successful_episode(session)

    assert session.is_batch_complete()

    assert (
        session.get_status()
        == SessionStatus.BATCH_COMPLETE
    )

    with pytest.raises(RuntimeError):
        session.create_episode()


def test_next_batch(session):

    for _ in range(3):
        complete_successful_episode(session)

    session.start_next_batch()

    assert session.batch_number == 2
    assert session.get_status() == SessionStatus.ACTIVE

    assert session.get_batch_statistics()["total"] == 0
    assert session.get_statistics()["total"] == 3


def test_end_with_unfinished_episode(session):

    session.create_episode()

    with pytest.raises(RuntimeError):
        session.end()


def test_end_session(session):

    session.end()

    assert session.get_status() == SessionStatus.COMPLETED
    assert session.ended_at is not None

    with pytest.raises(RuntimeError):
        session.create_episode()

    with pytest.raises(RuntimeError):
        session.start()

def test_config_hash(config):

    first = SessionManager(config)
    second = SessionManager(config)

    assert first.session_id != second.session_id
    assert first.config_hash == second.config_hash

    config.set("collection.random_seed", 100)

    third = SessionManager(config)

    assert first.config_hash == third.config_hash

    config.set("environment.width", 800)

    fourth = SessionManager(config)

    assert first.config_hash != fourth.config_hash


def test_seed_reproducibility(config):

    first = SessionManager(config)
    second = SessionManager(config)

    first.start()
    second.start()

    ep1 = first.create_episode()
    ep2 = second.create_episode()

    assert ep1.seed == ep2.seed

    assert (
        ep1.environment.get_state()
        == ep2.environment.get_state()
    )


def test_manifest_consistency(session):

    complete_successful_episode(session)

    session.end()

    path = session.session_dir / "session.yaml"

    with path.open() as file:
        manifest = yaml.safe_load(file)

    execution = manifest["execution"]

    assert execution == session.get_metadata()

    assert execution["statistics"]["successful"] == 1
    assert execution["status"] == "completed"

    assert len(execution["episodes"]) == 1

    episode_dir = (
        session.session_dir
        / execution["episodes"][0]["directory"]
    )

    assert episode_dir.exists()
    assert (episode_dir / "metadata.json").exists()

def test_discard_ready_episode(session):

    session.create_episode()
    session.discard_episode()

    assert session.current_episode is None
    assert session.get_statistics()["total"] == 0

    # Can create another episode
    episode = session.create_episode()

    assert episode.get_status() == EpisodeStatus.READY


def test_discard_running_episode(session):

    episode = session.create_episode()
    episode.start()

    with pytest.raises(RuntimeError):
        session.discard_episode()

    episode.abort()
    episode.finalize()
    session.complete_episode()


def test_discard_without_episode(session):

    with pytest.raises(RuntimeError):
        session.discard_episode()


def test_end_after_discard(session):

    session.create_episode()
    session.discard_episode()
    session.end()

    assert session.get_status() == SessionStatus.COMPLETED