import pytest

from src.core.config.config_manager import ConfigManager


@pytest.fixture
def config():
    return ConfigManager(
        "configs/experiment_001.yaml"
    )


def test_load(config):
    assert config.get("experiment.id") == "EXP001"


def test_get_nested_value(config):
    assert config.get("environment.width") == 640


def test_missing_key(config):
    assert config.get("unknown.key") is None


def test_default_value(config):
    assert config.get("unknown.key", 100) == 100


def test_set(config):
    config.set("operator.id", "operator_001")

    assert config.get("operator.id") == "operator_001"


def test_clone_isolation(config):
    cloned = config.clone()

    cloned.set("environment.width", 800)

    assert cloned.get("environment.width") == 800
    assert config.get("environment.width") == 640


def test_save_and_reload(config, tmp_path):
    path = tmp_path / "config.yaml"

    config.save(path)

    loaded = ConfigManager(path)

    assert loaded.to_dict() == config.to_dict()


def test_prevent_overwrite(config, tmp_path):
    path = tmp_path / "config.yaml"

    config.save(path)

    with pytest.raises(FileExistsError):
        config.save(path)


def test_invalid_radius(config):
    config.set(
        "environment.target.min_radius_px", 100
    )

    with pytest.raises(ValueError):
        config.validate()


def test_resolve_path(config):
    path = config.resolve_path(
        config.get("storage.raw_data_dir")
    )

    assert path.is_absolute()