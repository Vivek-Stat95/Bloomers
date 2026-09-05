"""
Hand-landmark gesture helpers, shared by either hand.
"""

import numpy as np

PINCH_THRESHOLD = 0.30  # lower = fingers must be closer together to count as a pinch


def true_handedness(label):
    """
    We flip the frame for a mirror/selfie view BEFORE running
    MediaPipe on it. MediaPipe's Left/Right label is based on the
    raw pixels it sees, so after flipping, a real right hand looks
    like a left hand to the model (and vice versa). This swaps the
    label back so "Right" really means the user's right hand,
    matching what they see mirrored on screen.
    """

    return "Left" if label == "Right" else "Right"


def get_bloom(hand):
    """
    Map hand openness (mean fingertip-to-wrist distance, normalized
    by palm width) to a bloom amount in [0, 1].
    """

    lm = hand.landmark

    wrist = np.array([lm[0].x, lm[0].y])
    index_mcp = np.array([lm[5].x, lm[5].y])
    pinky_mcp = np.array([lm[17].x, lm[17].y])

    palm_width = np.linalg.norm(index_mcp - pinky_mcp)

    fingertips = [8, 12, 16, 20]
    distances = [
        np.linalg.norm(np.array([lm[t].x, lm[t].y]) - wrist)
        for t in fingertips
    ]

    openness = np.mean(distances) / (palm_width + 1e-6)
    bloom = np.interp(openness, [1.20, 2.65], [0.0, 1.0])

    return float(np.clip(bloom, 0, 1))


def is_pinching(hand):
    """True when the thumb tip and index tip are close together."""

    lm = hand.landmark

    thumb_tip = np.array([lm[4].x, lm[4].y])
    index_tip = np.array([lm[8].x, lm[8].y])
    index_mcp = np.array([lm[5].x, lm[5].y])
    pinky_mcp = np.array([lm[17].x, lm[17].y])

    palm_width = np.linalg.norm(index_mcp - pinky_mcp)
    pinch_dist = np.linalg.norm(thumb_tip - index_tip) / (palm_width + 1e-6)

    return pinch_dist < PINCH_THRESHOLD
