import pygame
import numpy as np


class PygameRenderer:

    def __init__(self, config):

        self.width = config.get("environment.width")
        self.height = config.get("environment.height")

        self.background_color = config.get(
            "environment.background_color"
        )

        self.target_color = config.get(
            "environment.target.color"
        )

        self.cursor_color = config.get(
            "environment.cursor.color"
        )

        self.cursor_radius = config.get(
            "environment.cursor.radius_px", 4
        )

        self.surface = pygame.Surface(
            (self.width, self.height)
        )

    def render(self, state):

        self.surface.fill(self.background_color)

        pygame.draw.circle(
            self.surface,
            self.target_color,
            state.target_pos,
            state.target_radius
        )

        pygame.draw.circle(
            self.surface,
            self.cursor_color,
            tuple(round(v) for v in state.cursor_pos),
            self.cursor_radius
        )

    def get_observation(self):

        image = pygame.surfarray.array3d(self.surface)

        return np.ascontiguousarray(
            np.transpose(image, (1, 0, 2))
        )

    def get_surface(self):
        return self.surface