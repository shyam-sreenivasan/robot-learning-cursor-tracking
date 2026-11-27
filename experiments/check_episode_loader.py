from src.core.config.config_manager import ConfigManager

from src.data_factory.loaders.data_catalog import DataCatalog
from src.data_factory.loaders.episode_loader import EpisodeLoader


def main():

    config = ConfigManager(
        "configs/experiment_001.yaml"
    )

    catalog = DataCatalog(
        config.resolve_path(
            config.get("storage.raw_data_dir")
        )
    )

    loaded = 0
    failed = 0

    for session in catalog.list_sessions():

        session_id = session["session_id"]

        print(f"\nSession: {session_id}")

        for episode in catalog.list_episodes(session_id):

            episode_id = episode["episode_id"]

            try:

                loader = EpisodeLoader(
                    catalog,
                    session_id,
                    episode_id
                )

                metadata = loader.get_metadata()
                stats = loader.get_recording_stats()

                actions = loader.load_actions()
                states = loader.load_states()
                index = loader.load_observation_index()

                first_image = loader.load_observation(
                    index.iloc[0]["frame_id"]
                )

                last_image = loader.load_observation(
                    index.iloc[-1]["frame_id"]
                )

                print(
                    f"{episode_id[:8]} | "
                    f"{metadata['status']:8} | "
                    f"actions={len(actions):4} | "
                    f"states={len(states):4} | "
                    f"frames={len(index):4} | "
                    f"image={first_image.shape}"
                )

                loaded += 1

            except Exception as error:

                failed += 1

                print(
                    f"ERROR: {episode_id}: {error}"
                )

    print("\nLoading summary")
    print("Successfully loaded:", loaded)
    print("Loading failures:", failed)


if __name__ == "__main__":
    main()