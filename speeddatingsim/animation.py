"""
Speed dating circle animation
------------------------------
Two concentric rings of dots: an outer ring of red dots rotates around
a stationary inner ring of blue dots. Same number of dots on each ring.

Edit the CONFIG section below to make new versions (dot count, speed,
colors, sizes, duration, output format, etc).

Requires: matplotlib, ffmpeg (for mp4) or pillow (for gif)
Run: python3 speed_dating_animation.py
"""
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------------- CONFIG -----------------------
N_DOTS = 15          # number of dots per ring (same on both rings)
OUTER_R = 1.5        # outer ring radius
INNER_R = 1.1      # inner ring radius
DOT_SIZE = 260        # marker size (points^2)
OUTER_COLOR = "#E24B4A"   # red
INNER_COLOR = "#378ADD"   # blue
COLOURS = np.array(
    [
        INNER_COLOR,
        INNER_COLOR,
        INNER_COLOR,
        OUTER_COLOR,
        OUTER_COLOR,
        OUTER_COLOR,
        "#DD8A37",
        "#DD4BE2",
        "#318A37",
    ],
)

RANDOM_COLOURS = True
RANDOM_ASSIGNMENT = True

RING_LINE_COLOR = "#cccccc"

ROTATION_SPEED = 0.3   # radians per second for the outer ring
DURATION_SEC = 4 * np.pi / ROTATION_SPEED   # length of the animation
FPS = 10

# --------------------------------------------------------

n_frames = int(DURATION_SEC * FPS)

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect("equal")
ax.axis("off")

# guide rings (thin outline circles)
theta_full = np.linspace(0, 2 * np.pi, 200)
ax.plot((OUTER_R + INNER_R) / 2 * np.cos(theta_full), (OUTER_R + INNER_R) / 2 * np.sin(theta_full),
        color=RING_LINE_COLOR, linewidth=1, zorder=1)
# ax.plot(INNER_R * np.cos(theta_full), INNER_R * np.sin(theta_full),
#         color=RING_LINE_COLOR, linewidth=1, zorder=1)

inner_angles = np.arange(N_DOTS) * (2 * np.pi / N_DOTS)
inner_x = INNER_R * np.cos(inner_angles)
inner_y = INNER_R * np.sin(inner_angles)
outer_x = OUTER_R * np.cos(inner_angles)
outer_y = OUTER_R * np.sin(inner_angles)

lines = list(itertools.chain.from_iterable(
    ax.plot(
        (inner_x[i], outer_x[i]),
        (inner_y[i], outer_y[i]),
        "k-",
        lw=10,
    )
    for i in range(N_DOTS)
))

# stationary inner dots
if RANDOM_COLOURS:
    inner_colours = COLOURS[np.random.choice(len(COLOURS), N_DOTS)]
else:
    inner_colours = INNER_COLOR
inner_scatter = ax.scatter(inner_x, inner_y, s=DOT_SIZE, c=inner_colours,
                            zorder=3, edgecolors="white", linewidths=0.5)

# outer dots (positions updated each frame)
if RANDOM_COLOURS:
    outer_colours = COLOURS[np.random.choice(len(COLOURS), N_DOTS)]
else:
    outer_colours = OUTER_COLOR
outer_scatter = ax.scatter(outer_x, outer_y, s=DOT_SIZE, c=outer_colours,
                            zorder=3, edgecolors="white", linewidths=0.5)

matches = np.random.rand(N_DOTS, N_DOTS) > 0.9

inner_permutations = [
    np.random.permutation(N_DOTS)
    for _ in range(N_DOTS)
]
outer_permutations = [
    np.random.permutation(N_DOTS)
    for _ in range(N_DOTS)
]

def init():
    outer_scatter.set_offsets(np.empty((0, 2)))
    return outer_scatter,


def ease_sine(x: float) -> float:
    return -(np.cos(np.pi * x) - 1) / 2;

def update(frame):
    t = frame / FPS

    # rotates smoothly until the dots align, then pauses
    segment = 2 * np.pi / N_DOTS
    time_per_segment = (segment / ROTATION_SPEED)
    n_segments, unpaused = divmod(int(t / time_per_segment), 2)

    offset = (
        ease_sine((t % time_per_segment) / time_per_segment)
        if unpaused else 0
    )

    if RANDOM_ASSIGNMENT:
        inner_x0 = inner_x[inner_permutations[n_segments]]
        inner_y0 = inner_y[inner_permutations[n_segments]]
        inner_x1 = inner_x[inner_permutations[(n_segments + 1) % len(inner_permutations)]]
        inner_y1 = inner_y[inner_permutations[(n_segments + 1) % len(inner_permutations)]]
        this_inner_x = inner_x0 + (inner_x1 - inner_x0) * offset
        this_inner_y = inner_y0 + (inner_y1 - inner_y0) * offset
        inner_scatter.set_offsets(np.column_stack([this_inner_x, this_inner_y]))

        outer_x0 = outer_x[outer_permutations[n_segments]]
        outer_y0 = outer_y[outer_permutations[n_segments]]
        outer_x1 = outer_x[outer_permutations[(n_segments + 1) % len(outer_permutations)]]
        outer_y1 = outer_y[outer_permutations[(n_segments + 1) % len(outer_permutations)]]
        this_outer_x = outer_x0 + (outer_x1 - outer_x0) * offset
        this_outer_y = outer_y0 + (outer_y1 - outer_y0) * offset
        outer_scatter.set_offsets(np.column_stack([this_outer_x, this_outer_y]))

        line_colours = [
            ("#0000FF" if matches[inner_permutations[n_segments][i], outer_permutations[n_segments][i]] else "#FF0000")
            if not unpaused and t % time_per_segment > 0.33 else ("#FFFFFF", 0.0)
            for i in range(N_DOTS)
        ]

        for line_colour, line in zip(line_colours, lines):
            line.set_color(line_colour)

        return inner_scatter, outer_scatter, *lines
    else:
        base_angle = -(n_segments * segment + offset * segment)
        angles = base_angle + np.arange(N_DOTS) * (2 * np.pi / N_DOTS)
        x = OUTER_R * np.cos(angles)
        y = OUTER_R * np.sin(angles)
        outer_scatter.set_offsets(np.column_stack([x, y]))

        line_colours = [
            ("#0000FF" if matches[i, (i + n_segments) % N_DOTS] else "#FF0000")
            if not unpaused and t % time_per_segment > 0.33 else ("#FFFFFF", 0.0)
            for i in range(N_DOTS)
        ]

        for line_colour, line in zip(line_colours, lines):
            line.set_color(line_colour)

        return outer_scatter, *lines


ani = animation.FuncAnimation(
    fig, update, frames=n_frames, init_func=init,
    interval=1000 / FPS, blit=True
)

if __name__ == "__main__":
    import sys

    ani.save(sys.argv[1], writer=animation.PillowWriter(fps=FPS))
