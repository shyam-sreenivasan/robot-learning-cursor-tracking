from pathlib import Path
from copy import deepcopy

import yaml


class ConfigManager:

    def __init__(self, config_path=None, project_root=None):
        self.project_root = (
            Path(project_root).resolve()
            if project_root
            else Path(__file__).resolve().parents[3]
        )

        self._config = {}

        if config_path:
            self.load(config_path)

    def load(self, config_path):
        path = self.resolve_path(config_path)

        with path.open("r") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError("Configuration must be a mapping.")

        self._config = config
        self.validate()

        return self

    def get(self, key, default=None):
        value = self._config

        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default

            value = value[part]

        return deepcopy(value)

    def set(self, key, value):
        parts = key.split(".")
        current = self._config

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}

            if not isinstance(current[part], dict):
                raise ValueError(f"Invalid configuration path: {key}")

            current = current[part]

        current[parts[-1]] = deepcopy(value)

    def clone(self):
        cloned = ConfigManager(
            project_root=self.project_root
        )

        cloned._config = deepcopy(self._config)

        return cloned

    def save(self, config_path, overwrite=False):
        path = self.resolve_path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        mode = "w" if overwrite else "x"

        with path.open(mode) as file:
            yaml.safe_dump(
                self._config,
                file,
                sort_keys=False
            )

        return path

    def resolve_path(self, path):
        path = Path(path).expanduser()

        if not path.is_absolute():
            path = self.project_root / path

        return path.resolve()

    def to_dict(self):
        return deepcopy(self._config)

    def validate(self):
        required_sections = [
            "experiment",
            "storage",
            "environment",
            "collection",
            "operator"
        ]

        for section in required_sections:
            if not isinstance(self.get(section), dict):
                raise ValueError(
                    f"Missing or invalid section: {section}"
                )

        width = self.get("environment.width")
        height = self.get("environment.height")

        if not all(
            isinstance(v, int) and not isinstance(v, bool)
            and v > 0 for v in [width, height]
        ):
            raise ValueError("Invalid workspace dimensions.")

        min_radius = self.get(
            "environment.target.min_radius_px"
        )

        max_radius = self.get(
            "environment.target.max_radius_px"
        )

        if not all(
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            for v in [min_radius, max_radius]
        ):
            raise ValueError("Invalid target radius.")

        if not 0 < min_radius <= max_radius:
            raise ValueError("Invalid target radius range.")

        if round(max_radius) < 1 or (
            round(max_radius) * 2 >= min(width, height)
        ):
            raise ValueError("Invalid rendered target size.")

        positive_parameters = [
            "collection.render_hz",
            "collection.observation_hz",
            "collection.episode_timeout_sec",
            "collection.episodes_per_batch"
        ]

        for key in positive_parameters:
            value = self.get(key)

            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{key} must be positive.")

        batch_size = self.get(
            "collection.episodes_per_batch"
        )

        if not isinstance(batch_size, int):
            raise ValueError("Batch size must be an integer.")

        success_rate = self.get(
            "collection.expected_success_rate"
        )

        if not isinstance(success_rate, (int, float)) or not (
            0 <= success_rate <= 1
        ):
            raise ValueError("Invalid expected success rate.")

        return True