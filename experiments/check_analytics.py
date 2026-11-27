import json
from pprint import pprint

from src.data_factory.loaders.data_catalog import DataCatalog
from src.data_factory.loaders.episode_loader import EpisodeLoader

from src.data_factory.analytics.analytics import (
    get_overview,
    get_experiment_stats,
    get_session_stats,
    get_episode_summary,
    get_episode_trajectory
)


def main():

    catalog = DataCatalog("data/raw")

    # Global inventory

    print("\n========== GLOBAL OVERVIEW ==========")

    overview = get_overview(catalog)
    pprint(overview)

    json.dumps(overview, allow_nan=False)

    # Experiment analytics

    for experiment in catalog.list_experiments():

        result = get_experiment_stats(
            catalog,
            experiment["experiment_id"],
            experiment["config_hash"]
        )

        print("\n========== EXPERIMENT ==========")

        print("ID:", result["experiment_id"])
        print("Config:", result["config_hash"])
        print("Sessions:", result["total_sessions"])

        pprint(result["statistics"])

        coverage = result["coverage"]

        for name in ["cursor_coverage", "target_coverage"]:

            spatial = coverage[name]

            print(f"\n{name.upper()}")

            pprint(spatial["grid"])

            print(
                "Occupied bins:",
                spatial["occupied_bins"],
                "/",
                spatial["total_bins"]
            )

            print(
                "Coverage:",
                f"{spatial['coverage_ratio']:.2%}"
            )

            assert sum(
                sum(row) for row in spatial["grid"]
            ) == result["statistics"]["total_episodes"]

        json.dumps(result, allow_nan=False)

        print("\nExperiment checks: PASS")

    # Session analytics

    for session in catalog.list_sessions():

        result = get_session_stats(
            catalog,
            session["session_id"]
        )

        print("\n========== SESSION ==========")

        print("Session:", result["session_id"])
        print("Operator:", result["operator_id"])

        pprint(result["statistics"])

        assert (
            result["statistics"]["total_episodes"]
            == session["statistics"]["total"]
        )

        json.dumps(result, allow_nan=False)

        print("Session checks: PASS")

    # Individual episode analytics

    print("\n========== EPISODE ==========")

    for session in catalog.list_sessions():

        session_id = session["session_id"]
        episodes = catalog.list_episodes(session_id)

        if not episodes:
            continue

        loader = EpisodeLoader(
            catalog,
            session_id,
            episodes[0]["episode_id"]
        )

        summary = get_episode_summary(loader)
        trajectory = get_episode_trajectory(loader)

        pprint(summary)

        print("\nTrajectory samples:",
              len(trajectory["timestamps_sec"]))

        print("Initial timestamp:",
              trajectory["timestamps_sec"][0])

        print("Final timestamp:",
              trajectory["timestamps_sec"][-1])

        assert trajectory["timestamps_sec"][0] == 0

        assert (
            len(trajectory["timestamps_sec"])
            == len(trajectory["cursor_x"])
            == len(trajectory["cursor_y"])
        )

        json.dumps(summary, allow_nan=False)
        json.dumps(trajectory, allow_nan=False)

        print("Episode checks: PASS")

        break

    print("\nALL ANALYTICS CHECKS PASSED")


if __name__ == "__main__":
    main()