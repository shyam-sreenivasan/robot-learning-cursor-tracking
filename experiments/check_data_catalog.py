from pprint import pprint

from src.core.config.config_manager import ConfigManager
from src.data_factory.loaders.data_catalog import DataCatalog


config = ConfigManager(
    "configs/experiment_001.yaml"
)

root_dir = config.resolve_path(
    config.get("storage.raw_data_dir")
)

catalog = DataCatalog(root_dir)

print("\nEXPERIMENTS")
pprint(catalog.list_experiments())

print("\nSESSIONS")
sessions = catalog.list_sessions()
pprint(sessions)

for session in sessions:

    session_id = session["session_id"]

    episodes = catalog.list_episodes(session_id)

    print("\nSESSION:", session_id)
    print("Episode count:", len(episodes))

    if episodes:

        print("\nFirst episode:")
        pprint(episodes[0])