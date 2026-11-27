import pygame
import pytest

from src.interfaces.pygame_input_adapter import (
    PygameInputAdapter
)

from src.environments.action import Action, ButtonEvent


@pytest.fixture
def adapter():
    return PygameInputAdapter()


def test_mouse_movement(adapter):

    event = pygame.event.Event(
        pygame.MOUSEMOTION,
        rel=(10, -20)
    )

    action = adapter.translate(event)

    assert action == Action(10, -20)


def test_left_button_press(adapter):

    event = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN,
        button=1
    )

    action = adapter.translate(event)

    assert action.button_event == ButtonEvent.PRESS


def test_left_button_release(adapter):

    event = pygame.event.Event(
        pygame.MOUSEBUTTONUP,
        button=1
    )

    action = adapter.translate(event)

    assert action.button_event == ButtonEvent.RELEASE


def test_right_button_ignored(adapter):

    event = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN,
        button=3
    )

    assert adapter.translate(event) is None


def test_keyboard_event_ignored(adapter):

    event = pygame.event.Event(
        pygame.KEYDOWN,
        key=pygame.K_ESCAPE
    )

    assert adapter.translate(event) is None


def test_quit_event_ignored(adapter):

    event = pygame.event.Event(pygame.QUIT)

    assert adapter.translate(event) is None