import pygame

from src.data_acquisition.interfaces.pygame_input_adapter import PygameInputAdapter
from src.data_acquisition.interfaces.pygame_renderer import PygameRenderer

from src.data_acquisition.collection.data_contracts import (
    ActionRecord,
    ObservationRecord,
    StateRecord
)


class PygameApplication:

    def __init__(self, config, episode, recorder):

        self.config = config
        self.episode = episode
        self.recorder = recorder

        self.renderer = PygameRenderer(config)
        self.input_adapter = PygameInputAdapter()

        self.clock = episode.clock

        self.width = config.get("environment.width")
        self.height = config.get("environment.height")

        self.render_interval = (
            1 / config.get("collection.render_hz")
        )

        self.observation_interval = (
            1 / config.get("collection.observation_hz")
        )

        self.frame_id = 0
        self.event_id = 0

        self.screen = None

    def run(self, seed):

        pygame.init()

        try:
            self.screen = pygame.display.set_mode(
                (self.width, self.height)
            )

            pygame.display.set_caption("Mini Data Factory")

            self.episode.initialize(seed)

            pygame.event.set_grab(True)
            pygame.mouse.set_visible(False)
            pygame.event.clear()

            self._render()

            self.episode.start()

            self._capture_observation()

            next_render = (
                self.episode.started_at + self.render_interval
            )

            next_observation = (
                self.episode.started_at + self.observation_interval
            )

            loop_clock = pygame.time.Clock()

            while self.episode.is_running():

                self.episode.check_timeout()

                if not self.episode.is_running():
                    break

                self._process_events()

                if not self.episode.is_running():
                    break

                now = self.clock()

                if now >= next_observation:

                    self._capture_observation()

                    next_observation += self.observation_interval

                    # Skip missed sampling deadlines.
                    while next_observation <= now:
                        next_observation += self.observation_interval

                now = self.clock()

                if now >= next_render:

                    self._render()

                    next_render += self.render_interval

                    while next_render <= now:
                        next_render += self.render_interval

                loop_clock.tick(1000)

            self._finalize()

            return self.episode.get_metadata()

        finally:
            pygame.event.set_grab(False)
            pygame.mouse.set_visible(True)
            pygame.quit()

    def _process_events(self):

        for event in pygame.event.get():

            if not self.episode.is_running():
                break

            if event.type == pygame.QUIT:

                self.episode.abort()
                break

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:

                    self.episode.abort()
                    break

            action = self.input_adapter.translate(event)

            if action is not None:
                self._execute_action(action)

    def _execute_action(self, action):

        interaction = self.episode.step(action)

        if interaction is None:
            return

        self.recorder.record(
            "actions",
            ActionRecord(
                timestamp=interaction.timestamp,
                event_id=self.event_id,
                delta_x=action.delta_x,
                delta_y=action.delta_y,
                button_event=action.button_event.value
            )
        )

        self.event_id += 1

    def _capture_observation(self):

        state = self.episode.environment.get_state()

        timestamp = self.clock()

        self.renderer.render(state)
        image = self.renderer.get_observation()

        self.recorder.record(
            "observations",
            ObservationRecord(
                timestamp=timestamp,
                frame_id=self.frame_id,
                image=image
            )
        )

        self.recorder.record(
            "states",
            StateRecord(
                timestamp=timestamp,
                frame_id=self.frame_id,
                cursor_x=state.cursor_pos[0],
                cursor_y=state.cursor_pos[1],
                target_x=state.target_pos[0],
                target_y=state.target_pos[1],
                target_radius=state.target_radius,
                button_down=state.button_down
            )
        )

        self.frame_id += 1

    def _render(self):

        state = self.episode.environment.get_state()

        self.renderer.render(state)

        self.screen.blit(
            self.renderer.get_surface(),
            (0, 0)
        )

        pygame.display.flip()

    def _finalize(self):

        self._capture_observation()

        self.episode.finalize()