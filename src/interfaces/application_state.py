from enum import Enum


class ApplicationState(Enum):

    MAIN_MENU = "main_menu"
    COUNTDOWN = "countdown"
    RUNNING = "running"
    EPISODE_RESULT = "episode_result"
    BATCH_SUMMARY = "batch_summary"
    SESSION_SUMMARY = "session_summary"
    EXIT = "exit"