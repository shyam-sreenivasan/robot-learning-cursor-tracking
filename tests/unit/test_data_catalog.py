import pytest
import yaml

from src.data_factory.loaders.data_catalog import DataCatalog


def create_session(
    root_dir,
    session_id,
    experiment_id="EXP001",
    config_hash="hash_001",
    episode_count=2
):

    session_dir = root_dir / f"session_{session_id}"
    session_dir.mkdir()

    episodes = [
        {
            "episode_id": f"ep_{i}",
            "seed": i,
            "status": "success",
            "duration_sec": 2.0,
            "batch_number": 1,
            "termination_reason": "task_completed",
            "directory": f"episode_ep_{i}"
        }
        for i in range(episode_count)
    ]

    manifest = {
        "schema_version": "0.1",

        "experiment": {
            "id": experiment_id,
            "name": "stationary_target",
            "version": "0.1"
        },

        "operator": {
            "id": "operator_001"
        },

        "execution": {
            "session_id": session_id,
            "config_hash": config_hash,
            "status": "completed",
            "started_at": "2026-11-27T10:00:00+00:00",
            "ended_at": "2026-11-27T10:05:00+00:00",

            "statistics": {
                "total": episode_count,
                "successful": episode_count,
                "failed": 0,
                "aborted": 0,
                "evaluable": episode_count,
                "success_rate": (
                    1.0 if episode_count else None
                )
            },

            "episodes": episodes
        }
    }

    path = session_dir / "session.yaml"

    with path.open("w") as file:
        yaml.safe_dump(manifest, file)

    return session_dir


@pytest.fixture
def catalog(tmp_path):

    create_session(
        tmp_path, "001",
        episode_count=10
    )

    create_session(
        tmp_path, "002",
        episode_count=20
    )

    create_session(
        tmp_path, "003",
        config_hash="hash_002",
        episode_count=0
    )

    return DataCatalog(tmp_path)

def test_empty_catalog(tmp_path):

    catalog = DataCatalog(tmp_path)

    assert catalog.list_sessions() == []
    assert catalog.list_experiments() == []


def test_session_discovery(catalog):

    sessions = catalog.list_sessions()

    assert len(sessions) == 3

    assert {
        s["session_id"] for s in sessions
    } == {"001", "002", "003"}


def test_experiment_grouping(catalog):

    experiments = catalog.list_experiments()

    group = next(
        e for e in experiments
        if e["config_hash"] == "hash_001"
    )

    assert group["session_count"] == 2
    assert group["episode_count"] == 30


def test_different_configurations(catalog):

    experiments = catalog.list_experiments()

    assert len(experiments) == 2

    assert {
        e["config_hash"] for e in experiments
    } == {"hash_001", "hash_002"}


def test_session_filtering(catalog):

    sessions = catalog.list_sessions(
        experiment_id="EXP001",
        config_hash="hash_001"
    )

    assert len(sessions) == 2

    assert catalog.list_sessions(
        experiment_id="UNKNOWN"
    ) == []


def test_episode_discovery(catalog):

    episodes = catalog.list_episodes("001")

    assert len(episodes) == 10
    assert episodes[0]["episode_id"] == "ep_0"

    # An unregistered directory should not
    # appear in catalog results.

    orphan = (
        catalog.root_dir
        / "session_001"
        / "episode_unregistered"
    )

    orphan.mkdir()

    assert len(
        catalog.list_episodes("001")
    ) == 10


def test_empty_session(catalog):

    assert catalog.list_episodes("003") == []


def test_get_episode(catalog):

    episode = catalog.get_episode(
        "001", "ep_0"
    )

    assert episode["episode_id"] == "ep_0"
    assert episode["status"] == "success"


def test_unknown_session(catalog):

    with pytest.raises(FileNotFoundError):
        catalog.get_session("unknown")


def test_unknown_episode(catalog):

    with pytest.raises(KeyError):
        catalog.get_episode("001", "unknown")


def test_malformed_manifest(tmp_path):

    session_dir = tmp_path / "session_invalid"
    session_dir.mkdir()

    path = session_dir / "session.yaml"
    path.write_text("- invalid\n- manifest")

    catalog = DataCatalog(tmp_path)

    with pytest.raises(ValueError):
        catalog.list_sessions()


def test_read_only_access(catalog):

    paths = list(
        catalog.root_dir.rglob("session.yaml")
    )

    before = {
        path: (
            path.read_bytes(),
            path.stat().st_mtime_ns
        )
        for path in paths
    }

    catalog.list_experiments()
    catalog.list_sessions()
    catalog.get_session("001")
    catalog.list_episodes("001")
    catalog.get_episode("001", "ep_0")

    after = {
        path: (
            path.read_bytes(),
            path.stat().st_mtime_ns
        )
        for path in paths
    }

    assert before == after

