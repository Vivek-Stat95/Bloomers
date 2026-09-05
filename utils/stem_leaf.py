"""
Stem + leaf drawing, shared by both flowers.

Drawn directly in screen space (not part of the 3-D flower math)
since they're just decorative and should stay perfectly fixed below
each flower - unaffected by bloom or hand tracking.
"""

import math

import cv2
import numpy as np

STEM_COLOR = (34, 139, 34)      # forest green, BGR
LEAF_COLOR = (34, 120, 34)      # slightly darker green, BGR
LEAF_VEIN_COLOR = (20, 90, 20)  # darker still, for the center vein

STEM_TOP_OFFSET = 0    # how far below the flower's anchor the stem starts
STEM_SWAY = 22          # gentle S-curve amount

LEAF_T1 = 0.42
LEAF_LENGTH1 = 95
LEAF_WIDTH1 = 40
LEAF_ANGLE_DEG1 = 200   # direction the leaf points (0 = +x/right, 180 = left)

LEAF_T2 = 0.62
LEAF_LENGTH2 = 105
LEAF_WIDTH2 = 45
LEAF_ANGLE_DEG2 = 320


def _leaf_polygon(attach, length, width, angle_deg):
    """
    Almond-shaped leaf outline: width(u) = (W/2) sin(pi u) along the
    leaf's length, rotated by angle_deg and translated to `attach`.
    """

    u = np.linspace(0.0, 1.0, 24)
    w = (width / 2.0) * np.sin(np.pi * u)

    top = np.stack([u * length, w], axis=1)
    bottom = np.stack([(u * length)[::-1], -w[::-1]], axis=1)
    poly = np.vstack([top, bottom])

    theta = math.radians(angle_deg)
    rot = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ])

    poly = poly @ rot.T
    poly[:, 0] += attach[0]
    poly[:, 1] += attach[1]

    return poly.astype(np.int32)


def draw_stem_and_leaf(frame, anchor):
    """Draw a gently curved stem with two leaves below `anchor`."""

    top_x, top_y = anchor[0], anchor[1] + STEM_TOP_OFFSET
    bottom_y = frame.shape[0] - 30

    t = np.linspace(0.0, 1.0, 40)
    stem_x = top_x + STEM_SWAY * np.sin(np.pi * t)
    stem_y = top_y + t * (bottom_y - top_y)

    stem_pts = np.stack([stem_x, stem_y], axis=1).astype(np.int32)

    cv2.polylines(frame, [stem_pts], False, STEM_COLOR, 6, cv2.LINE_AA)

    for leaf_t, leaf_len, leaf_w, leaf_ang in (
        (LEAF_T1, LEAF_LENGTH1, LEAF_WIDTH1, LEAF_ANGLE_DEG1),
        (LEAF_T2, LEAF_LENGTH2, LEAF_WIDTH2, LEAF_ANGLE_DEG2),
    ):
        idx = int(leaf_t * (len(stem_pts) - 1))
        attach = stem_pts[idx]

        leaf_poly = _leaf_polygon(attach, leaf_len, leaf_w, leaf_ang)
        cv2.fillPoly(frame, [leaf_poly], LEAF_COLOR)

        theta = math.radians(leaf_ang)
        tip = (
            int(attach[0] + leaf_len * math.cos(theta)),
            int(attach[1] + leaf_len * math.sin(theta))
        )

        cv2.line(frame, (int(attach[0]), int(attach[1])), tip, LEAF_VEIN_COLOR, 2, cv2.LINE_AA)
