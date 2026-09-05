"""
Flower 2: Violet (blooms with the LEFT hand).

Same surface family as the rose, but theta only sweeps one loop
(no spiral) and phi is a CONSTANT rather than theta-dependent,
which is what removes the progressive/spiralling unfurl effect.
See MATH.md for the full derivation.
"""

import math

import numpy as np

from utils.common import build_triangle_grid, render_surface, view_rotation_matrix_y_up

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

A2 = 1.95653
B2 = 1.27689
PETAL_NUM2 = 5

N_R2 = 60
N_THETA2 = 220

# ------------------------------------------------------------
# Grid (r spans 0..1, theta spans a single loop: 0..2*pi)
# ------------------------------------------------------------

R2G, THETA2G = np.meshgrid(
    np.linspace(0, 1, N_R2),
    np.linspace(0, 2 * np.pi, N_THETA2),
    indexing="ij"
)

X_TERM2 = (
    1
    - 0.5
    * (
        (5 / 4) * (1 - np.mod(PETAL_NUM2 * THETA2G, 2 * np.pi) / np.pi) ** 2
        - 1 / 4
    ) ** 2
)

# phi at full bloom = the reference plot's constant phi.
# (pi/2)*exp(-2*pi/(8*pi)) simplifies to (pi/2)*exp(-0.25)
PHI_OPEN2 = (math.pi / 2) * math.exp(-0.25)
# phi at zero bloom = a near-flat, unopened bud
PHI_CLOSED2 = 0.05

ROWS2, COLS2 = R2G.shape
TRIANGLES = build_triangle_grid(ROWS2, COLS2)

# Fixed camera view (Y-up), close to the reference plot's
# view([-12.700 81.200]) - a near top-down angle. See MATH.md
# section 5.
VIEW_R = view_rotation_matrix_y_up(elev_deg=70.0, azim_deg=-12.7)

# Color LUT: reference violet_map = [gold_map; blue_map]
#   gold_map: 2 rows, (255,215,0) -> (250,210,0)
#   blue_map: 20 rows, blueviolet (138,43,226) -> indigo (75,0,130)
# Resampled to a smooth 256-entry ramp, keyed off radial distance
# from center so the flower's core stays gold and the petal tips
# fade to indigo.
LUT_SIZE = 256

_gold_rows = np.array([
    [255, 215, 0],
    [250, 210, 0],
]) / 255.0

_blue_map_size = 20
_blue_rows = np.stack([
    np.linspace(138, 75, _blue_map_size),
    np.linspace(43, 0, _blue_map_size),
    np.linspace(226, 130, _blue_map_size),
], axis=1) / 255.0

_violet_map = np.vstack([_gold_rows, _blue_rows])  # 22 rows, low->high

_src_idx = np.linspace(0.0, 1.0, len(_violet_map))
_dst_idx = np.linspace(0.0, 1.0, LUT_SIZE)

LUT = np.stack([
    np.interp(_dst_idx, _src_idx, _violet_map[:, ch])
    for ch in range(3)
], axis=1)

# Radial coordinate (0 at center, 1 at petal tip) - drives the
# color gradient, precomputed once since it never changes.
R_FLAT = R2G.ravel()


def create_violet(bloom):
    """Build the (X, Y, Z) violet surface for a given bloom amount (0..1)."""

    phi = PHI_CLOSED2 + bloom * (PHI_OPEN2 - PHI_CLOSED2)

    y = A2 * (R2G ** 2) * (B2 * R2G - 1) ** 2 * np.sin(phi)
    r2 = X_TERM2 * (R2G * np.sin(phi) + y * np.cos(phi))

    X = r2 * np.sin(THETA2G)
    Y = r2 * np.cos(THETA2G)
    Z = X_TERM2 * (R2G * np.cos(phi) - y * np.sin(phi))

    return X, Y, Z


def draw_violet(frame, bloom, center):
    """Render the violet at the given bloom amount, anchored at `center`."""

    X, Y, Z = create_violet(bloom)
    render_surface(
        frame, X, Y, Z,
        TRIANGLES, LUT, LUT_SIZE, R_FLAT,
        VIEW_R, center
    )
