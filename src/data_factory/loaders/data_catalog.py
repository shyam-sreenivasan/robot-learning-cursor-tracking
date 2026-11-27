from pathlib import Path
from copy import deepcopy

import yaml


class DataCatalog:

    def __init__(self, root_dir):

        self.root_dir = Path(root_dir).resolve()

        if not self.root_dir.is_dir():
            raise FileNotFoundError(self.root_dir)

    def list_sessions(
        self,
        experiment_id=None,
        config_hash=None
    ):

        sessions = []

        for path in sorted(
            self.root_dir.glob("session_*/session.yaml")
        ):

            manifest = self._read_manifest(path)

            experiment = manifest["experiment"]
            execution = manifest["execution"]

            if experiment_id is not None:

                if experiment["id"] != experiment_id:
                    continue

            if config_hash is not None:

                if execution["config_hash"] != config_hash:
                    continue

            sessions.append({
                "session_id": execution["session_id"],
                "experiment_id": experiment["id"],
                "experiment_name": experiment["name"],
                "experiment_version": experiment["version"],
                "config_hash": execution["config_hash"],
                "operator_id": manifest.get(
                    "operator", {}
                ).get("id"),
                "status": execution["status"],
                "started_at": execution["started_at"],
                "ended_at": execution["ended_at"],
                "statistics": execution["statistics"]
            })

        return sessions

    def list_experiments(self):

        experiments = {}

        for session in self.list_sessions():

            key = (
                session["experiment_id"],
                session["config_hash"]
            )

            if key not in experiments:

                experiments[key] = {
                    "experiment_id":
                        session["experiment_id"],
                    "name":
                        session["experiment_name"],
                    "version":
                        session["experiment_version"],
                    "config_hash":
                        session["config_hash"],
                    "session_count": 0,
                    "episode_count": 0
                }

            experiment = experiments[key]

            experiment["session_count"] += 1

            experiment["episode_count"] += (
                session["statistics"]["total"]
            )

        return list(experiments.values())

    def get_session(self, session_id):

        if (
            not session_id
            or Path(session_id).name != session_id
        ):
            raise ValueError("Invalid session ID.")

        path = (
            self.root_dir
            / f"session_{session_id}"
            / "session.yaml"
        )

        manifest = self._read_manifest(path)

        if manifest["execution"]["session_id"] != session_id:
            raise ValueError("Session identity mismatch.")

        return manifest

    def list_episodes(self, session_id):

        manifest = self.get_session(session_id)

        return deepcopy(
            manifest["execution"]["episodes"]
        )

    def get_episode(self, session_id, episode_id):

        for episode in self.list_episodes(session_id):

            if episode["episode_id"] == episode_id:
                return episode

        raise KeyError(
            f"Episode not found: {episode_id}"
        )

    @staticmethod
    def _read_manifest(path):

        with path.open("r") as file:

            manifest = yaml.safe_load(file)

        if not isinstance(manifest, dict):
            raise ValueError(
                f"Invalid manifest: {path}"
            )

        return manifest