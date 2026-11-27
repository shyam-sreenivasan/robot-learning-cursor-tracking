from src.config.config_manager import ConfigManager
from src.interfaces.pygame_application import PygameApplication


def main():

    config = ConfigManager(
        "configs/experiment_001.yaml"
    )

    app = PygameApplication(config)
    app.run()


if __name__ == "__main__":
    main()