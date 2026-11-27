import pygame
import time
import numpy as np
import matplotlib.pyplot as plt


pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Mouse Sampling Test")

timestamps = []
clock = pygame.time.Clock()

duration = 10
loop_hz = 1000
start = time.perf_counter()

running = True
displacements = []
while running and time.perf_counter() - start < duration:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEMOTION:
            timestamps.append(time.perf_counter())
                        # Inside MOUSEMOTION event handler
            dx, dy = event.rel
            displacements.append(np.hypot(dx, dy))

    clock.tick(loop_hz)

pygame.quit()
print("Mean displacement per event:",
      np.mean(displacements))

print("Total distance (px):",
      np.sum(displacements))
if len(timestamps) > 1:
    intervals = np.diff(timestamps)

    print("Total events:", len(timestamps))
    print("Mean frequency (Hz):", 1 / intervals.mean())
    print("Mean interval (ms):", intervals.mean() * 1000)
    print("Median interval (ms):", np.median(intervals) * 1000)
    print("Std deviation (ms):", intervals.std() * 1000)
    print("Max interval (ms):", intervals.max() * 1000)

    intervals_ms = intervals * 1000

    plt.hist(intervals_ms[intervals_ms < 50], bins=100)
    plt.xlabel("Inter-event interval (ms)")
    plt.ylabel("Frequency")
    plt.title("Mouse Event Interval Distribution (< 50 ms)")
    plt.grid(alpha=0.3)
    plt.show()

    for p in [5, 25, 50, 75, 90, 95, 99]:
        print(f"P{p}: {np.percentile(intervals_ms, p):.2f} ms")