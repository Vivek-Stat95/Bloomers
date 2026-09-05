"""
Two-Hand Flower Bloom
======================

Right hand -> Rose (left side of screen)
Left hand  -> Violet (right side of screen)

Gesture flow, per hand:
  1. Pinch thumb + index to "plant" that flower's stem/leaf.
  2. Open your hand to bloom the flower.
  3. If that hand leaves the frame, its flower disappears (state is
     kept, so it resumes right where it left off when the hand
     returns).

Both flowers bloom simultaneously when both hands are on screen.

See MATH.md for the full mathematical derivation of both flower
surfaces, the camera transform, and the hand-gesture mappings.
"""

import cv2
import mediapipe as mp

from utils.rose import draw_rose
from utils.violet import draw_violet
from utils.hand_gestures import true_handedness, get_bloom, is_pinching
from utils.stem_leaf import draw_stem_and_leaf


def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Per-flower state: each hand/flower tracks its own "has the
    # stem been planted" and "current bloom amount" independently.
    rose_stem_planted = False
    rose_smooth_bloom = 0.0

    violet_stem_planted = False
    violet_smooth_bloom = 0.0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Rose: left side of screen. Violet: right side of screen.
        rose_center = (int(frame.shape[1] * 0.28), int(frame.shape[0] * 0.55))
        violet_center = (int(frame.shape[1] * 0.72), int(frame.shape[0] * 0.55))

        right_hand = None
        left_hand = None

        if result.multi_hand_landmarks and result.multi_handedness:
            for landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = true_handedness(handedness.classification[0].label)

                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                if label == "Right":
                    right_hand = landmarks
                else:
                    left_hand = landmarks

        # ------------------------------------------------------
        # ROSE - right hand only
        # ------------------------------------------------------
        if right_hand is not None:
            if not rose_stem_planted and is_pinching(right_hand):
                rose_stem_planted = True

            if rose_stem_planted:
                target_bloom = get_bloom(right_hand)
                rose_smooth_bloom = 0.82 * rose_smooth_bloom + 0.18 * target_bloom

            if rose_stem_planted:
                draw_stem_and_leaf(frame, rose_center)
                draw_rose(frame, rose_smooth_bloom, rose_center)
        # else: right hand not visible -> rose disappears this frame.
        # rose_stem_planted / rose_smooth_bloom are kept as-is, so it
        # resumes right where it left off once the right hand returns.

        # ------------------------------------------------------
        # VIOLET - left hand only
        # ------------------------------------------------------
        if left_hand is not None:
            if not violet_stem_planted and is_pinching(left_hand):
                violet_stem_planted = True

            if violet_stem_planted:
                target_bloom2 = get_bloom(left_hand)
                violet_smooth_bloom = 0.82 * violet_smooth_bloom + 0.18 * target_bloom2

            if violet_stem_planted:
                draw_stem_and_leaf(frame, violet_center)
                draw_violet(frame, violet_smooth_bloom, violet_center)

        # ============================================================
        # UI
        # ============================================================

        y_line = 45

        if right_hand is not None:
            text = f"ROSE (right hand)  BLOOM {int(rose_smooth_bloom * 100)}%" \
                if rose_stem_planted else "Right hand: pinch to plant the rose's stem"
        else:
            text = "Right hand not detected - rose hidden"

        cv2.putText(frame, text, (30, y_line), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        y_line += 35

        if left_hand is not None:
            text2 = f"VIOLET (left hand)  BLOOM {int(violet_smooth_bloom * 100)}%" \
                if violet_stem_planted else "Left hand: pinch to plant the violet's stem"
        else:
            text2 = "Left hand not detected - violet hidden"

        cv2.putText(frame, text2, (30, y_line), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Two-Hand Flower Bloom", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
