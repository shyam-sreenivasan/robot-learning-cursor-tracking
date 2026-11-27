
import math
import time

import pygame

from .application_state import ApplicationState
from .pygame_input_adapter import PygameInputAdapter
from .pygame_renderer import PygameRenderer

from src.collection.session_manager import SessionManager

from src.collection.data_contracts import (
    ActionRecord,
    ObservationRecord,
    StateRecord
)


class PygameApplication:

    MENU_WIDTH = 760
    MENU_HEIGHT = 680
    RESULT_DURATION = 1.5

    EDITABLE_PARAMETERS = [
        ("Operator ID", "operator.id", str),
        ("Workspace width", "environment.width", int),
        ("Workspace height", "environment.height", int),
        ("Min target radius",
         "environment.target.min_radius_px", float),
        ("Max target radius",
         "environment.target.max_radius_px", float),
        ("Timeout (sec)",
         "collection.episode_timeout_sec", float),
        ("Render Hz", "collection.render_hz", int),
        ("Observation Hz", "collection.observation_hz", int),
        ("Episodes per batch",
         "collection.episodes_per_batch", int),
        ("Countdown (sec)",
         "collection.countdown_sec", int)
    ]

    def __init__(self, config):

        self.config = config.clone()

        self.state = ApplicationState.MAIN_MENU

        self.session = None
        self.episode = None
        self.recorder = None

        self.renderer = None
        self.input_adapter = PygameInputAdapter()

        self.screen = None
        self.font = None
        self.title_font = None
        self.large_font = None
        self.small_font = None
        self.clock = None

        self.width = self.MENU_WIDTH
        self.height = self.MENU_HEIGHT

        self.background_color = (25, 25, 25)
        self.text_color = (230, 230, 230)
        self.secondary_color = (160, 160, 160)

        self.button_color = (40, 100, 200)
        self.success_color = (80, 200, 120)
        self.failure_color = (230, 90, 90)

        self.fields = {}
        self.active_field = None
        self.error_message = ""

        self.frame_id = 0
        self.event_id = 0

        self.next_render_at = None
        self.next_observation_at = None

        self.render_interval = None
        self.observation_interval = None

        self.countdown_ends_at = None
        self.result_ends_at = None

        self.last_result = None

        self.start_button = pygame.Rect(
            480, 615, 250, 42
        )

        self.continue_button = pygame.Rect(
            100, 475, 560, 48
        )

        self.end_button = pygame.Rect(
            100, 540, 560, 48
        )

        self.menu_button = pygame.Rect(
            100, 540, 560, 48
        )

        self._initialize_fields()

    # ==================================================
    # APPLICATION LIFECYCLE
    # ==================================================

    def run(self):

        pygame.init()

        self._set_menu_display()

        pygame.display.set_caption("Mini Data Factory")

        self.font = pygame.font.Font(None, 26)
        self.small_font = pygame.font.Font(None, 21)
        self.title_font = pygame.font.Font(None, 38)
        self.large_font = pygame.font.Font(None, 100)

        self.clock = pygame.time.Clock()

        try:

            while self.state != ApplicationState.EXIT:

                self._handle_events()

                if self.state == ApplicationState.EXIT:
                    break

                self._update()

                if self.state == ApplicationState.RUNNING:

                    # Rendering is independently scheduled.
                    self.clock.tick(1000)

                elif self.state != ApplicationState.EXIT:

                    self._draw()
                    self.clock.tick(60)

        finally:

            pygame.event.set_grab(False)
            pygame.mouse.set_visible(True)

            pygame.quit()

    def _set_menu_display(self):

        self.width = self.MENU_WIDTH
        self.height = self.MENU_HEIGHT

        self.screen = pygame.display.set_mode(
            (self.width, self.height)
        )

    def _set_episode_display(self):

        self.width = self.session.config.get(
            "environment.width"
        )

        self.height = self.session.config.get(
            "environment.height"
        )

        self.screen = pygame.display.set_mode(
            (self.width, self.height)
        )

    # ==================================================
    # EVENT HANDLING
    # ==================================================

    def _handle_events(self):

        for event in pygame.event.get():

            previous_state = self.state

            if event.type == pygame.QUIT:
                self._exit()
                break

            if self.state == ApplicationState.MAIN_MENU:
                self._handle_menu_event(event)

            elif self.state == ApplicationState.COUNTDOWN:
                self._handle_countdown_event(event)

            elif self.state == ApplicationState.RUNNING:
                self._handle_running_event(event)

            elif self.state == ApplicationState.EPISODE_RESULT:
                self._handle_result_event(event)

            elif self.state == ApplicationState.BATCH_SUMMARY:
                self._handle_batch_event(event)

            elif self.state == ApplicationState.SESSION_SUMMARY:
                self._handle_summary_event(event)

            # Do not process old queued events in a new state.
            if self.state != previous_state:
                break

    def _handle_menu_event(self, event):

        if event.type == pygame.MOUSEBUTTONDOWN:

            if event.button != 1:
                return

            if self.start_button.collidepoint(event.pos):
                self._start_session()
                return

            self.active_field = None

            for key, field in self.fields.items():

                if field["rect"].collidepoint(event.pos):
                    self.active_field = key
                    break

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                self._exit()
                return

            if self.active_field is None:
                return

            field = self.fields[self.active_field]

            if event.key == pygame.K_BACKSPACE:

                field["value"] = field["value"][:-1]

            elif event.key == pygame.K_RETURN:

                self.active_field = None

            elif event.unicode.isprintable():

                if len(field["value"]) < 40:
                    field["value"] += event.unicode

    def _handle_countdown_event(self, event):

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:

                self.session.discard_episode()
                self.episode = None
                self.recorder = None

                self._end_session()

    def _handle_running_event(self, event):

        if not self.episode.is_running():
            return

        self.episode.check_timeout()

        if self.episode.is_terminal():
            self._finish_episode()
            return

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:

                self.episode.abort()
                self._finish_episode()
                return

        action = self.input_adapter.translate(event)

        if action is not None:
            self._execute_action(action)

        if self.episode.is_terminal():
            self._finish_episode()

    def _handle_result_event(self, event):

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                self._end_session()

    def _handle_batch_event(self, event):

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                self._end_session()

        elif event.type == pygame.MOUSEBUTTONDOWN:

            if event.button != 1:
                return

            if self.continue_button.collidepoint(event.pos):

                self.session.start_next_batch()
                self._prepare_episode()

            elif self.end_button.collidepoint(event.pos):

                self._end_session()

    def _handle_summary_event(self, event):

        if event.type == pygame.KEYDOWN:

            if event.key in (
                pygame.K_ESCAPE,
                pygame.K_RETURN
            ):
                self._return_to_menu()

        elif event.type == pygame.MOUSEBUTTONDOWN:

            if event.button == 1:

                if self.menu_button.collidepoint(event.pos):
                    self._return_to_menu()

    # ==================================================
    # APPLICATION STATE UPDATES
    # ==================================================

    def _update(self):

        if self.state == ApplicationState.COUNTDOWN:

            if time.monotonic() >= self.countdown_ends_at:
                self._start_episode()

        elif self.state == ApplicationState.RUNNING:

            self._update_running_episode()

        elif self.state == ApplicationState.EPISODE_RESULT:

            if time.monotonic() >= self.result_ends_at:

                if self.session.is_batch_complete():

                    self.state = (
                        ApplicationState.BATCH_SUMMARY
                    )

                else:

                    self._prepare_episode()

    def _update_running_episode(self):

        self.episode.check_timeout()

        if self.episode.is_terminal():
            self._finish_episode()
            return

        now = self.episode.clock()

        if now >= self.next_observation_at:

            self._capture_observation()

            self.next_observation_at = (
                self._advance_deadline(
                    self.next_observation_at,
                    self.observation_interval,
                    self.episode.clock()
                )
            )

        self.episode.check_timeout()

        if self.episode.is_terminal():
            self._finish_episode()
            return

        now = self.episode.clock()

        if now >= self.next_render_at:

            self._render_workspace()

            self.next_render_at = (
                self._advance_deadline(
                    self.next_render_at,
                    self.render_interval,
                    self.episode.clock()
                )
            )

    @staticmethod
    def _advance_deadline(deadline, interval, now):

        while deadline <= now:
            deadline += interval

        return deadline

    # ==================================================
    # CONFIGURATION AND SESSION MANAGEMENT
    # ==================================================

    def _initialize_fields(self):

        y = 95

        for label, key, converter in self.EDITABLE_PARAMETERS:

            value = self.config.get(key)

            self.fields[key] = {
                "label": label,
                "converter": converter,
                "value": "" if value is None else str(value),
                "rect": pygame.Rect(390, y, 340, 30)
            }

            y += 38

    def _apply_configuration(self):

        candidate = self.config.clone()

        for key, field in self.fields.items():

            value = field["value"].strip()
            converter = field["converter"]

            if key == "operator.id":

                candidate.set(key, value or None)
                continue

            if not value:
                raise ValueError(
                    f"{field['label']} is required."
                )

            converted = converter(value)

            if not math.isfinite(converted):

                raise ValueError(
                    f"Invalid value for {field['label']}."
                )

            candidate.set(key, converted)

        candidate.validate()

        countdown = candidate.get(
            "collection.countdown_sec"
        )

        if countdown < 0:
            raise ValueError(
                "Countdown cannot be negative."
            )

        width = candidate.get("environment.width")
        height = candidate.get("environment.height")

        if not (
            100 <= width <= 1920
            and 100 <= height <= 1080
        ):

            raise ValueError(
                "Workspace must be between "
                "100x100 and 1920x1080."
            )

        self.config = candidate

    def _start_session(self):

        self.error_message = ""

        try:

            self._apply_configuration()

        except (ValueError, TypeError, OverflowError) as error:

            self.error_message = str(error)
            return

        self.active_field = None

        self.session = SessionManager(self.config)

        self.session.start()

        print("Session started:", self.session.session_id)
        print("Directory:", self.session.session_dir)

        self._prepare_episode()

    def _end_session(self):

        self.session.end()

        print("\nSession completed:")
        print(self.session.get_statistics())

        self.state = ApplicationState.SESSION_SUMMARY

        self._set_menu_display()

    def _return_to_menu(self):

        self.session = None
        self.episode = None
        self.recorder = None
        self.last_result = None

        self.active_field = None
        self.error_message = ""

        self.state = ApplicationState.MAIN_MENU

        self._set_menu_display()

    # ==================================================
    # EPISODE LIFECYCLE
    # ==================================================

    def _prepare_episode(self):

        self.episode = self.session.create_episode()

        self.recorder = self.episode.recorder

        self.renderer = PygameRenderer(
            self.session.config
        )

        self.frame_id = 0
        self.event_id = 0

        self.render_interval = (
            1 / self.session.config.get(
                "collection.render_hz"
            )
        )

        self.observation_interval = (
            1 / self.session.config.get(
                "collection.observation_hz"
            )
        )

        countdown = self.session.config.get(
            "collection.countdown_sec"
        )

        self.countdown_ends_at = (
            time.monotonic() + countdown
        )

        self.state = ApplicationState.COUNTDOWN

    def _start_episode(self):

        self._set_episode_display()

        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

        # Discard inputs generated outside the episode.
        pygame.event.clear([
            pygame.MOUSEMOTION,
            pygame.MOUSEBUTTONDOWN,
            pygame.MOUSEBUTTONUP
        ])

        self.episode.start()

        self.state = ApplicationState.RUNNING

        self._render_workspace()

        # Explicit initial observation.
        self._capture_observation()

        self.next_render_at = (
            self._advance_deadline(
                self.episode.started_at,
                self.render_interval,
                self.episode.clock()
            )
        )

        self.next_observation_at = (
            self._advance_deadline(
                self.episode.started_at,
                self.observation_interval,
                self.episode.clock()
            )
        )

    def _finish_episode(self):

        if not self.episode.is_terminal():
            raise RuntimeError(
                "Cannot finalize a running episode."
            )

        # Explicit terminal observation.
        self._capture_observation()

        # Stop recorder only after final data delivery.
        self.episode.finalize()

        self.last_result = self.session.complete_episode()

        print("\nEpisode completed:")
        print(self.last_result)

        print("Recording statistics:")
        print(self.recorder.get_stats())

        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)

        pygame.event.clear([
            pygame.MOUSEMOTION,
            pygame.MOUSEBUTTONDOWN,
            pygame.MOUSEBUTTONUP
        ])

        self.episode = None
        self.recorder = None

        self._set_menu_display()

        self.result_ends_at = (
            time.monotonic() + self.RESULT_DURATION
        )

        self.state = ApplicationState.EPISODE_RESULT

    # ==================================================
    # DATA ACQUISITION
    # ==================================================

    def _execute_action(self, action):

        if not self.episode.is_running():
            return

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

        timestamp = self.episode.clock()

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

    def _render_workspace(self):

        state = self.episode.environment.get_state()

        self.renderer.render(state)

        self.screen.blit(
            self.renderer.get_surface(),
            (0, 0)
        )

        pygame.display.flip()

    # ==================================================
    # GRACEFUL APPLICATION EXIT
    # ==================================================

    def _exit(self):

        if self.state == ApplicationState.RUNNING:

            self.episode.check_timeout()

            if self.episode.is_running():
                self.episode.abort()

            self._finish_episode()

        elif self.state == ApplicationState.COUNTDOWN:

            self.session.discard_episode()

            self.episode = None
            self.recorder = None

        if self.session is not None:

            if self.session.get_status().value != "completed":
                self.session.end()

        self.state = ApplicationState.EXIT

    # ==================================================
    # COMMON UI HELPERS
    # ==================================================

    def _draw(self):

        self.screen.fill(self.background_color)

        if self.state == ApplicationState.MAIN_MENU:
            self._draw_main_menu()

        elif self.state == ApplicationState.COUNTDOWN:
            self._draw_countdown()

        elif self.state == ApplicationState.EPISODE_RESULT:
            self._draw_episode_result()

        elif self.state == ApplicationState.BATCH_SUMMARY:
            self._draw_batch_summary()

        elif self.state == ApplicationState.SESSION_SUMMARY:
            self._draw_session_summary()

        pygame.display.flip()

    def _draw_text(
        self,
        text,
        position,
        font=None,
        color=None
    ):

        font = font or self.font
        color = color or self.text_color

        surface = font.render(
            str(text), True, color
        )

        self.screen.blit(surface, position)

    def _draw_centered(
        self,
        text,
        y,
        font=None,
        color=None
    ):

        font = font or self.font
        color = color or self.text_color

        surface = font.render(
            str(text), True, color
        )

        rect = surface.get_rect(
            center=(self.width // 2, y)
        )

        self.screen.blit(surface, rect)

    def _draw_button(self, rect, label):

        hovered = rect.collidepoint(
            pygame.mouse.get_pos()
        )

        color = (
            (60, 120, 220)
            if hovered else self.button_color
        )

        pygame.draw.rect(
            self.screen,
            color,
            rect,
            border_radius=6
        )

        surface = self.font.render(
            label, True, (255, 255, 255)
        )

        self.screen.blit(
            surface,
            surface.get_rect(center=rect.center)
        )

    def _draw_progress(self, completed, total, y):

        ratio = min(
            completed / max(total, 1),
            1.0
        )

        background = pygame.Rect(
            100, y, 560, 12
        )

        pygame.draw.rect(
            self.screen,
            (65, 65, 65),
            background,
            border_radius=5
        )

        if ratio > 0:

            progress = pygame.Rect(
                100,
                y,
                int(560 * ratio),
                12
            )

            pygame.draw.rect(
                self.screen,
                self.success_color,
                progress,
                border_radius=5
            )

    def _draw_wrapped_text(
        self,
        text,
        x,
        y,
        max_width
    ):

        words = text.split()
        line = ""

        for word in words:

            candidate = f"{line} {word}".strip()

            if self.small_font.size(candidate)[0] > max_width:

                self._draw_text(
                    line,
                    (x, y),
                    self.small_font
                )

                y += 22
                line = word

            else:

                line = candidate

        if line:

            self._draw_text(
                line,
                (x, y),
                self.small_font
            )

    # ==================================================
    # MAIN MENU
    # ==================================================

    def _draw_main_menu(self):

        self._draw_text(
            "Mini Data Factory",
            (30, 20),
            self.title_font
        )

        self._draw_text(
            self.config.get("experiment.name"),
            (30, 58),
            color=self.secondary_color
        )

        for key, field in self.fields.items():

            rect = field["rect"]
            active = self.active_field == key

            self._draw_text(
                field["label"],
                (30, rect.y + 5)
            )

            pygame.draw.rect(
                self.screen,
                (42, 42, 42),
                rect,
                border_radius=4
            )

            border_color = (
                (80, 150, 240)
                if active else (85, 85, 85)
            )

            pygame.draw.rect(
                self.screen,
                border_color,
                rect,
                width=2,
                border_radius=4
            )

            value = field["value"]

            if active:
                value += "|"

            text_surface = self.font.render(
                value,
                True,
                self.text_color
            )

            previous_clip = self.screen.get_clip()
            self.screen.set_clip(rect.inflate(-12, -4))

            self.screen.blit(
                text_surface,
                (rect.x + 8, rect.y + 5)
            )

            self.screen.set_clip(previous_clip)

        self._draw_text(
            "Instructions",
            (30, 485),
            color=self.secondary_color
        )

        instructions = self.config.get(
            "operator.instructions", ""
        )

        self._draw_wrapped_text(
            instructions,
            30, 515, 700
        )

        if self.error_message:

            self._draw_text(
                self.error_message,
                (30, 590),
                self.small_font,
                self.failure_color
            )

        self._draw_button(
            self.start_button,
            "Start Session"
        )

        self._draw_text(
            "ESC to exit",
            (30, 630),
            color=self.secondary_color
        )

    # ==================================================
    # COUNTDOWN SCREEN
    # ==================================================

    def _draw_countdown(self):

        stats = self.session.get_batch_statistics()

        number = stats["evaluable"] + 1

        remaining = max(
            0,
            math.ceil(
                self.countdown_ends_at
                - time.monotonic()
            )
        )

        self._draw_centered(
            f"Batch {self.session.batch_number}",
            100,
            self.title_font
        )

        self._draw_centered(
            f"Episode {number} of "
            f"{self.session.batch_size}",
            155
        )

        self._draw_centered(
            "Get ready",
            240,
            color=self.secondary_color
        )

        self._draw_centered(
            remaining,
            345,
            self.large_font
        )

        self._draw_centered(
            "ESC to end session",
            600,
            color=self.secondary_color
        )

    # ==================================================
    # EPISODE RESULT SCREEN
    # ==================================================

    def _draw_episode_result(self):

        result = self.last_result

        status = result["status"]

        if status == "success":

            label = "SUCCESS"
            color = self.success_color

        elif status == "failed":

            label = "TIMEOUT"
            color = self.failure_color

        else:

            label = "ABORTED"
            color = self.secondary_color

        self._draw_centered(
            label,
            170,
            self.large_font,
            color
        )

        self._draw_centered(
            f"Duration: {result['duration_sec']:.2f} sec",
            260
        )

        stats = self.session.get_batch_statistics()

        completed = stats["evaluable"]
        total = self.session.batch_size

        self._draw_centered(
            f"{completed} of {total} "
            "evaluable episodes completed",
            340
        )

        self._draw_progress(
            completed, total, 380
        )

        if self.session.is_batch_complete():

            message = "Batch complete"

        else:

            message = "Next episode begins shortly..."

        self._draw_centered(
            message,
            460,
            color=self.secondary_color
        )

        self._draw_centered(
            "ESC to end session",
            600,
            color=self.secondary_color
        )

    # ==================================================
    # STATISTICS SCREENS
    # ==================================================

    def _draw_statistics(self, stats):

        rows = [
            ("Recorded episodes", stats["total"]),
            ("Successful", stats["successful"]),
            ("Failed", stats["failed"]),
            ("Aborted", stats["aborted"]),
            ("Evaluable episodes", stats["evaluable"])
        ]

        y = 140

        for label, value in rows:

            self._draw_text(label, (140, y))
            self._draw_text(value, (530, y))

            y += 42

        rate = stats["success_rate"]

        rate_text = (
            "N/A"
            if rate is None
            else f"{rate * 100:.1f}%"
        )

        self._draw_centered(
            f"Success rate: {rate_text}",
            390,
            self.title_font,
            self.success_color
        )

    def _draw_batch_summary(self):

        self._draw_centered(
            f"Batch {self.session.batch_number} Complete",
            65,
            self.title_font
        )

        stats = self.session.get_batch_statistics()

        self._draw_statistics(stats)

        self._draw_button(
            self.continue_button,
            "Continue Next Batch"
        )

        self._draw_button(
            self.end_button,
            "End Session"
        )

    def _draw_session_summary(self):

        self._draw_centered(
            "Session Complete",
            65,
            self.title_font
        )

        self._draw_statistics(
            self.session.get_statistics()
        )

        self._draw_centered(
            "All finalized episodes saved.",
            460,
            color=self.secondary_color
        )

        self._draw_button(
            self.menu_button,
            "Return to Main Menu"
        )
