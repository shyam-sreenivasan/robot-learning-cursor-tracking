import math
import random

import pygame

from src.core.environments.environment import Environment
from src.core.tasks.click_target_task import ClickTargetTask
from src.data_acquisition.collection.episode import Episode

from src.evaluation.evaluation import (
    SimulatedClock,
    build_observation,
    execute_prediction
)

from src.policies.behavior_cloning import predict_action


# ============================================================
# EPISODE INITIALIZATION
# ============================================================

def create_live_episode(config, seed):

    clock = SimulatedClock()

    environment = Environment(config)
    task = ClickTargetTask(config)

    episode = Episode(
        environment,
        task,
        config,
        clock=clock
    )

    episode.initialize(seed)
    episode.start()

    return episode, clock


# ============================================================
# RENDERING
# ============================================================

def render_live_episode(
    screen,
    font,
    episode,
    trajectory,
    prediction,
    seed,
    paused,
    checkpoint_name,
    click_attempts,
    click_threshold
):

    environment = episode.environment
    state = environment.get_state()

    width = environment.width
    height = environment.height

    screen.fill((20, 20, 20))

    workspace = screen.subsurface(
        pygame.Rect(0, 0, width, height)
    )

    workspace.fill((25, 25, 25))

    # Draw trajectory.

    if len(trajectory) > 1:

        pygame.draw.lines(
            workspace,
            (70, 130, 210),
            False,
            trajectory,
            2
        )

    # Draw target.

    pygame.draw.circle(
        workspace,
        (0, 255, 0),
        state.target_pos,
        state.target_radius
    )

    # Draw cursor.

    cursor = tuple(
        round(v) for v in state.cursor_pos
    )

    pygame.draw.circle(
        workspace,
        (255, 255, 255),
        cursor,
        5
    )

    # Draw predicted displacement, exaggerated
    # for visibility.

    if prediction:

        endpoint = (
            cursor[0] + prediction["delta_x"] * 10,
            cursor[1] + prediction["delta_y"] * 10
        )

        pygame.draw.line(
            workspace,
            (255, 180, 0),
            cursor,
            endpoint,
            2
        )

    # Telemetry panel.

    pygame.draw.rect(
        screen,
        (36, 40, 46),
        (width, 0, 310, height)
    )

    distance = math.dist(
        state.cursor_pos,
        state.target_pos
    )

    status = episode.get_status().value

    if paused and episode.is_running():
        status = "paused"

    lines = [
        ("POLICY EXECUTION", (255, 255, 255)),
        ("", None),
        (f"Model: {checkpoint_name}", None),
        (f"Seed: {seed}", None),
        (f"Status: {status.upper()}", None),
        (f"Time: {episode.get_duration():.2f} s", None),
        (f"Distance: {distance:.2f} px", None),
        (f"Target radius: {state.target_radius} px", None),
        (f"Click attempts: {click_attempts}", None),
        ("", None),
        ("POLICY OUTPUT", (255, 255, 255))
    ]

    if prediction:

        lines.extend([
            (
                f"Delta X: {prediction['delta_x']:.3f}",
                None
            ),
            (
                f"Delta Y: {prediction['delta_y']:.3f}",
                None
            ),
            (
                f"Click probability: "
                f"{prediction['click_probability']:.4f}",
                None
            ),
            (
                f"Click threshold: {click_threshold:.2f}",
                None
            ),
            (
                f"Click: {prediction['click']}",
                None
            )
        ])

    else:

        lines.append(("Waiting for prediction", None))

    lines.extend([
        ("", None),
        ("CONTROLS", (255, 255, 255)),
        ("SPACE - Pause / Resume", None),
        ("R - Restart same scenario", None),
        ("N - New random scenario", None),
        ("ESC - Exit", None)
    ])

    y = 15

    for text, color in lines:

        surface = font.render(
            text,
            True,
            color or (190, 195, 205)
        )

        screen.blit(
            surface,
            (width + 15, y)
        )

        y += 20

    pygame.display.flip()


# ============================================================
# LIVE EXECUTION LOOP
# ============================================================

def run_live_policy(
    model,
    environment_config,
    dataset_manifest,
    inference_config,
    checkpoint_name,
    seed=None
):

    pygame.init()

    width = environment_config.get(
        "environment.width"
    )

    height = environment_config.get(
        "environment.height"
    )

    screen = pygame.display.set_mode(
        (width + 310, height)
    )

    pygame.display.set_caption(
        "Mini Data Factory - Live Policy Evaluation"
    )

    font = pygame.font.SysFont("monospace", 14)

    render_clock = pygame.time.Clock()

    decision_hz = inference_config["decision_hz"]
    click_threshold = inference_config["click_threshold"]

    if decision_hz <= 0:
        raise ValueError("Invalid decision frequency.")

    if seed is None:
        seed = random.randrange(2**32)

    episode, clock = create_live_episode(
        environment_config,
        seed
    )

    trajectory = [
        episode.environment.get_state().cursor_pos
    ]

    prediction = None
    click_attempts = 0

    step = 0
    paused = False
    running = True

    try:

        while running:

            restart = False

            # Handle user controls.

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        paused = not paused

                    elif event.key == pygame.K_r:
                        restart = True

                    elif event.key == pygame.K_n:

                        seed = random.randrange(2**32)
                        restart = True

            if not running:
                break

            if restart:

                if episode.is_running():
                    episode.abort()
                    episode.finalize()

                episode, clock = create_live_episode(
                    environment_config,
                    seed
                )

                trajectory = [
                    episode.environment.get_state().cursor_pos
                ]

                prediction = None
                click_attempts = 0

                step = 0
                paused = False

            # Execute one policy decision.

            if not paused and episode.is_running():

                clock.set_time(
                    min(
                        step / decision_hz,
                        episode.timeout
                    )
                )

                if not episode.check_timeout():

                    state = episode.environment.get_state()

                    observation = build_observation(
                        state,
                        dataset_manifest
                    )

                    prediction = predict_action(
                        model,
                        observation,
                        click_threshold=click_threshold
                    )

                    if not all(
                        math.isfinite(prediction[key])
                        for key in [
                            "delta_x",
                            "delta_y",
                            "click_probability"
                        ]
                    ):
                        raise ValueError(
                            "Non-finite policy prediction."
                        )

                    interaction = execute_prediction(
                        episode,
                        prediction
                    )

                    if interaction is not None:

                        trajectory.append(
                            interaction.current_state.cursor_pos
                        )

                        if interaction.task_result.attempted:
                            click_attempts += 1

                    step += 1

                if episode.is_terminal():

                    episode.finalize()

                    print("\nEPISODE COMPLETED")
                    print(episode.get_metadata())
                    print("Click attempts:", click_attempts)

            # Render current state.

            render_live_episode(
                screen,
                font,
                episode,
                trajectory,
                prediction,
                seed,
                paused,
                checkpoint_name,
                click_attempts,
                click_threshold
            )

            render_clock.tick(decision_hz)

    finally:

        if episode.is_running():
            episode.abort()
            episode.finalize()

        pygame.quit()