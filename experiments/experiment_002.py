import argparse
import time

import numpy as np
import pygame

parser = argparse.ArgumentParser()
parser.add_argument('--fps', type=int, choices=[0, 30, 60], default=30)
parser.add_argument('--duration', type=float, default=10)
args = parser.parse_args()

pygame.init()
WIDTH, HEIGHT = 640, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(f'Experiment 002 | capture {args.fps} Hz | SPACE to start')
font = pygame.font.Font(None, 28)
clock = pygame.time.Clock()
pygame.mouse.set_visible(False)


def draw(cursor):
    screen.fill((25, 30, 40))
    pygame.draw.circle(screen, (70, 210, 110), (480, 240), 20)
    pygame.draw.circle(screen, (255, 255, 255), cursor, 5)


waiting = True
while waiting:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            waiting = False
    draw(pygame.mouse.get_pos())
    screen.blit(font.render('Press SPACE, then move continuously', True, (255, 255, 255)), (15, 15))
    pygame.display.flip()
    clock.tick(60)

pygame.event.clear()
cursor = pygame.mouse.get_pos()
frames, frame_times, frame_slots = [], [], []
event_times, distances, capture_ms, loop_times, render_times = [], [], [], [], []
max_batch = 0
render_slot = 0
start = time.perf_counter_ns()
end = start + int(args.duration * 1e9)
running = True

while running:
    now = time.perf_counter_ns()
    if now >= end:
        break
    loop_times.append(now)
    batch = 0
    for event in pygame.event.get():
        received = time.perf_counter_ns()
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        elif event.type == pygame.MOUSEMOTION:
            event_times.append(received)
            distances.append(float(np.hypot(*event.rel)))
            cursor = event.pos
            batch += 1
    max_batch = max(max_batch, batch)
    if not running:
        break

    now = time.perf_counter_ns()
    # All conditions render at 60 Hz. Capture every second render at 30 Hz.
    due_slot = (now - start) * 60 // 1_000_000_000
    if due_slot >= render_slot:
        draw(cursor)
        if args.fps and (args.fps == 60 or due_slot % 2 == 0):
            t0 = time.perf_counter_ns()
            frame = pygame.surfarray.array3d(screen).transpose(1, 0, 2).copy()
            t1 = time.perf_counter_ns()
            frames.append(frame)
            frame_times.append(t0)
            frame_slots.append(due_slot if args.fps == 60 else due_slot // 2)
            capture_ms.append((t1 - t0) / 1e6)
        pygame.display.flip()
        render_times.append(time.perf_counter_ns())
        render_slot = due_slot + 1
    clock.tick(1000)

elapsed = (time.perf_counter_ns() - start) / 1e9
pygame.quit()


def intervals(label, times):
    if len(times) < 2:
        print(f'{label}: insufficient samples')
        return
    dt = np.diff(np.asarray(times, dtype=np.int64)) / 1e6
    print(f'{label} intervals (ms): median={np.median(dt):.3f}, '
          f'p95={np.percentile(dt, 95):.3f}, std={dt.std():.3f}, max={dt.max():.3f}')


print(f'\nCapture setting: {args.fps} Hz | elapsed: {elapsed:.3f} s | aborted: {not running}')
print(f'Mouse events: {len(event_times)} | rate: {len(event_times)/elapsed:.2f} Hz')
print(f'Cursor speed (px/s): {sum(distances)/elapsed:.2f}')
print(f'Max motion events per queue drain: {max_batch}')
intervals('Mouse reception', event_times)
print(f'Loop rate: {len(loop_times)/elapsed:.2f} Hz')
intervals('Loop', loop_times)
print(f'Render rate: {len(render_times)/elapsed:.2f} Hz')
intervals('Render', render_times)
if args.fps:
    expected = int(np.ceil(min(elapsed, args.duration) * args.fps))
    print(f'Frames: {len(frames)} | rate: {len(frames)/elapsed:.2f} Hz')
    print(f'Expected scheduled frames (approx): {expected} | missed: {max(0, expected-len(frames))}')
    intervals('Frame capture', frame_times)
    if capture_ms:
        print(f'Capture time (ms): mean={np.mean(capture_ms):.3f}, p95={np.percentile(capture_ms,95):.3f}, max={max(capture_ms):.3f}')
    print(f'Raw frame memory (MiB): {sum(f.nbytes for f in frames)/2**20:.2f}')
