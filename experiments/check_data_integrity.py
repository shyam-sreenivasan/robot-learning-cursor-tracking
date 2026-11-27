from collections import Counter

from src.core.config.config_manager import ConfigManager
from src.data_factory.loaders.data_catalog import DataCatalog
from src.data_factory.loaders.episode_loader import EpisodeLoader

from src.data_factory.validation.episode_integrity_validator import (
    EpisodeIntegrityValidator
)


config = ConfigManager("configs/experiment_001.yaml")

catalog = DataCatalog(
    config.resolve_path(
        config.get("storage.raw_data_dir")
    )
)

summary = Counter()
check_summary = {}


for session in catalog.list_sessions():

    session_id = session["session_id"]

    for episode in catalog.list_episodes(session_id):

        episode_id = episode["episode_id"]

        try:

            loader = EpisodeLoader(
                catalog, session_id, episode_id
            )

            validator = EpisodeIntegrityValidator(loader)

            results = validator.validate()

        except Exception as error:

            summary["fail"] += 1

            print(f"{episode_id}: ERROR {error}")

            continue

        statuses = {r.status.value for r in results}

        if "fail" in statuses:
            status = "fail"
        elif "skipped" in statuses:
            status = "incomplete"
        else:
            status = "pass"

        summary[status] += 1

        print(f"{episode_id}: {status.upper()}")

        for result in results:

            check_summary.setdefault(
                result.check_name, Counter()
            )[result.status.value] += 1

            if result.status.value != "pass":

                print("  Check:", result.check_name)
                print("  Message:", result.message)
                print("  Expected:", result.expected)
                print("  Observed:", result.observed)


print("\nEPISODE VALIDATION SUMMARY")

for status, count in summary.items():
    print(status, count)

print("\nINDIVIDUAL CHECK SUMMARY")

for name, counts in check_summary.items():
    print(name, dict(counts))