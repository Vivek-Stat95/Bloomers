"""
Flower 1: Rose (blooms with the RIGHT hand).

Surface equations match the reference matplotlib plot exactly at
full bloom. See MATH.md for the full derivation.
"""

import numpy as np

from utils.common import build_triangle_grid, render_surface, view_rotation_matrix_y_up

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

A = 1.995653
B = 1.27689
C = 8
PETAL_NUM = 3.6

N_R = 60
N_THETA = 220

# ------------------------------------------------------------
# Grid (r spans 0..1, theta spans a spiral: -2 .. 20*pi)
# ------------------------------------------------------------

R, THETA = np.meshgrid(
    np.linspace(0, 1, N_R),
    np.linspace(-2, 20 * np.pi, N_THETA),
    indexing="ij"
)

# Petal shaping term - identical to the reference plot's "x" term,
# precomputed once since it never changes with bloom.
X_TERM = (
    1
    - 0.5
    * (
        (5 / 4) * (1 - np.mod(PETAL_NUM * THETA, 2 * np.pi) / np.pi) ** 2
        - 1 / 4
    ) ** 2
)

# phi at full bloom = exactly the reference plot's phi (theta-dependent,
# which is what gives the rose its progressive, spiralling unfurl)
PHI_OPEN = (np.pi / 2) * np.exp(-THETA / (C * np.pi))
# phi at zero bloom = a near-flat, unopened bud
PHI_CLOSED = 0.05 * np.exp(-THETA / (C * np.pi))

ROWS, COLS = R.shape
TRIANGLES = build_triangle_grid(ROWS, COLS)

# Fixed camera view (Y-up). See MATH.md section 5.
VIEW_R = view_rotation_matrix_y_up(elev_deg=18.0, azim_deg=-25.0)

# Color LUT: red (1,0,0) -> dark red (0.25,0,0)
LUT_SIZE = 256
_red_vals = np.linspace(1.0, 0.25, LUT_SIZE)
LUT = np.zeros((LUT_SIZE, 3))
LUT[:, 0] = _red_vals

# Radial coordinate (0 at center, 1 at petal tip) - drives the
# color gradient, precomputed once since it never changes.
R_FLAT = R.ravel()


def create_rose(bloom):
    """Build the (X, Y, Z) rose surface for a given bloom amount (0..1)."""

    phi = PHI_CLOSED + bloom * (PHI_OPEN - PHI_CLOSED)

    y = A * (R ** 2) * (B * R - 1) ** 2 * np.sin(phi)
    r2 = X_TERM * (R * np.sin(phi) + y * np.cos(phi))

    X = r2 * np.sin(THETA)
    Y = r2 * np.cos(THETA)
    Z = X_TERM * (R * np.cos(phi) - y * np.sin(phi))

    return X, Y, Z


def draw_rose(frame, bloom, center):
    """Render the rose at the given bloom amount, anchored at `center`."""

    X, Y, Z = create_rose(bloom)
    render_surface(
        frame, X, Y, Z,
        TRIANGLES, LUT, LUT_SIZE, R_FLAT,
        VIEW_R, center
    )
