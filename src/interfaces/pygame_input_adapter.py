import pygame

from src.environments.action import Action, ButtonEvent


class PygameInputAdapter:

    def translate(self, event):

        if event.type == pygame.MOUSEMOTION:

            dx, dy = event.rel

            return Action(
                delta_x=dx,
                delta_y=dy
            )

        if event.type == pygame.MOUSEBUTTONDOWN:

            if event.button == 1:
                return Action(
                    button_event=ButtonEvent.PRESS
                )

        if event.type == pygame.MOUSEBUTTONUP:

            if event.button == 1:
                return Action(
                    button_event=ButtonEvent.RELEASE
                )

        return None