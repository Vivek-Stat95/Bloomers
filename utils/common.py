"""
Shared rendering helpers used by every flower.

Both flowers are generated as a (r, theta) -> (X, Y, Z) surface where
Z is the "bloom/opening" axis. This module provides the pieces every
flower needs afterwards: a fixed Y-up camera matrix, a perspective
projector, and a generic depth-sorted, gradient-shaded triangle
renderer.
"""

import math

import cv2
import numpy as np


def view_rotation_matrix_y_up(elev_deg, azim_deg):
    """
    Build a fixed camera rotation matrix that treats Y as "up".

    Each flower's own create_XXX(bloom) function returns Z as the
    bloom/opening axis. render_surface() remaps that onto world +Y
    before applying this matrix, so azimuth spins around Y (the
    flower's opening direction) and elevation tilts around X.

    This is a constant, fixed viewing angle - it is NOT driven by
    the hand, so the flower never rotates on screen.
    """

    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)

    # azimuth: spin around the up axis (Y)
    Ry = np.array([
        [ math.cos(azim), 0, math.sin(azim)],
        [ 0,               1, 0             ],
        [-math.sin(azim), 0, math.cos(azim)]
    ])

    # elevation: tilt around X
    Rx = np.array([
        [1, 0,               0             ],
        [0, math.cos(elev), -math.sin(elev)],
        [0, math.sin(elev),  math.cos(elev)]
    ])

    return Rx @ Ry


def project(points, center, scale=170, focal=650):
    """
    Simple perspective projection: points (N,3) in camera space ->
    (px, py) pixel coordinates around a fixed screen anchor `center`.
    """

    depth = focal / (focal + points[:, 2] * 100)

    px = (center[0] + points[:, 0] * scale * depth).astype(np.int32)
    py = (center[1] - points[:, 1] * scale * depth).astype(np.int32)

    return px, py


def render_surface(frame, X, Y, Z, triangles, lut, lut_size, grad_flat, view_r, center):
    """
    Generic renderer shared by every flower.

    Convention: X, Y, Z come from a create_XXX(bloom) function where
    Z is the "bloom/opening" axis. This:
      1. Remaps Z onto world +Y (X stays X, old Y becomes depth) so
         every flower opens "upward" the same way.
      2. Applies the flower's own fixed camera matrix (view_r).
      3. Flips depth so the bloom opens toward the camera.
      4. Depth-sorts triangles (painter's algorithm) and fills each
         one, shaded by a blend of `grad_flat` (typically the radial
         coordinate - distance from the flower's center) and
         normalized camera depth, looked up in `lut`.
    """

    pts = np.stack([X.ravel(), Z.ravel(), Y.ravel()], axis=1)

    pts = pts @ view_r.T
    pts[:, 2] = -pts[:, 2]  # bloom opens toward the camera, not away

    px, py = project(pts, center)
    z_view = pts[:, 2]

    z_min, z_max = z_view.min(), z_view.max()
    z_norm = (z_view - z_min) / (z_max - z_min + 1e-8)

    tri_depth = z_view[triangles].mean(axis=1)
    draw_order = np.argsort(-tri_depth)  # painter's algorithm: far first

    for idx in draw_order:
        tri = triangles[idx]
        poly = np.stack([px[tri], py[tri]], axis=1).astype(np.int32)

        gradient = grad_flat[tri].mean()
        depth_shade = z_norm[tri].mean()
        shade = 0.75 * gradient + 0.25 * depth_shade

        lut_idx = int(np.clip(shade * (lut_size - 1), 0, lut_size - 1))
        r_val, g_val, b_val = lut[lut_idx]

        color = (int(b_val * 255), int(g_val * 255), int(r_val * 255))  # BGR
        cv2.fillConvexPoly(frame, poly, color)


def build_triangle_grid(rows, cols):
    """
    Precompute the triangle index topology for an (rows x cols)
    parametric grid. Every flower's grid never changes shape once
    built, so this only needs to run once per flower.
    """

    tris = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            p1 = i * cols + j
            p2 = i * cols + j + 1
            p3 = (i + 1) * cols + j
            p4 = (i + 1) * cols + j + 1
            tris.append((p1, p2, p3))
            tris.append((p2, p4, p3))

    return np.array(tris)
