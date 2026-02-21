

import cv2
import numpy as np
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.core.base_options import BaseOptions
import mediapipe as mp
import os

# Download the hand landmark model if not present
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("Downloading hand_landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Helper to draw overlay
def draw_custom_overlay(image, landmarks):
    h, w, _ = image.shape
    points = []
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
    if len(points) < 21:
        return

    # Only draw if two hands are detected (handled in main loop)
    pass

def draw_finger_strings_between_hands(image, hand1, hand2):
    # Draw neon lines between corresponding fingertips of both hands, with a glowing circle at each midpoint
    h, w, _ = image.shape
    neon_glow = (255, 255, 100)
    neon_mid = (255, 255, 180)
    neon_core = (255, 255, 255)
    tip_indices = [4, 8, 12, 16, 20]
    pts1 = [(int(hand1[i].x * w), int(hand1[i].y * h)) for i in tip_indices]
    pts2 = [(int(hand2[i].x * w), int(hand2[i].y * h)) for i in tip_indices]
    for p1, p2 in zip(pts1, pts2):
        # Draw glowing line between fingertips
        overlay = image.copy()
        cv2.line(overlay, p1, p2, neon_glow, 12, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
        overlay2 = image.copy()
        cv2.line(overlay2, p1, p2, neon_mid, 6, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay2, 0.35, image, 0.65, 0, image)
        cv2.line(image, p1, p2, neon_core, 2, lineType=cv2.LINE_AA)
        # Draw glowing circle at the midpoint, radius proportional to string length
        mx, my = int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)
        dist = int(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
        base_radius = 10
        radius = max(base_radius, int(dist * 0.10))  # 10% of string length, min 10px
        overlay3 = image.copy()
        cv2.circle(overlay3, (mx, my), radius+8, neon_glow, -1)
        cv2.addWeighted(overlay3, 0.18, image, 0.82, 0, image)
        overlay4 = image.copy()
        cv2.circle(overlay4, (mx, my), radius, neon_mid, -1)
        cv2.addWeighted(overlay4, 0.25, image, 0.75, 0, image)
        cv2.circle(image, (mx, my), max(7, radius//2), neon_core, 2, lineType=cv2.LINE_AA)
        cv2.circle(image, (mx, my), radius, neon_core, 1, lineType=cv2.LINE_AA)

# Initialize hand landmarker
options = hand_landmarker.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)
landmarker = hand_landmarker.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    if result.hand_landmarks:
        # Only proceed if two hands are detected
        if len(result.hand_landmarks) == 2:
            # Assign left/right hands by x-coordinate of wrist (landmark 0)
            hand0 = result.hand_landmarks[0]
            hand1 = result.hand_landmarks[1]
            h, w, _ = frame.shape
            x0 = hand0[0].x * w
            x1 = hand1[0].x * w
            if x0 < x1:
                left_hand, right_hand = hand0, hand1
            else:
                left_hand, right_hand = hand1, hand0

            # Smoothing: keep a short history of fingertip positions and average them
            if not hasattr(draw_finger_strings_between_hands, 'tip_history'):
                draw_finger_strings_between_hands.tip_history = []
            tip_indices = [4, 8, 12, 16, 20]
            left_tips = [(left_hand[i].x, left_hand[i].y) for i in tip_indices]
            right_tips = [(right_hand[i].x, right_hand[i].y) for i in tip_indices]
            draw_finger_strings_between_hands.tip_history.append((left_tips, right_tips))
            # Keep only last 3 frames for smoothing
            if len(draw_finger_strings_between_hands.tip_history) > 3:
                draw_finger_strings_between_hands.tip_history.pop(0)
            # Average positions
            avg_left = [(
                sum(t[0][i][0] for t in draw_finger_strings_between_hands.tip_history)/len(draw_finger_strings_between_hands.tip_history),
                sum(t[0][i][1] for t in draw_finger_strings_between_hands.tip_history)/len(draw_finger_strings_between_hands.tip_history)
            ) for i in range(5)]
            avg_right = [(
                sum(t[1][i][0] for t in draw_finger_strings_between_hands.tip_history)/len(draw_finger_strings_between_hands.tip_history),
                sum(t[1][i][1] for t in draw_finger_strings_between_hands.tip_history)/len(draw_finger_strings_between_hands.tip_history)
            ) for i in range(5)]
            # Create fake landmark objects for smoothed tips
            class LM: pass
            left_hand_smoothed = list(left_hand)
            right_hand_smoothed = list(right_hand)
            for idx, (lx, ly) in zip(tip_indices, avg_left):
                lm = LM(); lm.x = lx; lm.y = ly; lm.z = 0
                left_hand_smoothed[idx] = lm
            for idx, (rx, ry) in zip(tip_indices, avg_right):
                lm = LM(); lm.x = rx; lm.y = ry; lm.z = 0
                right_hand_smoothed[idx] = lm
            draw_finger_strings_between_hands(frame, left_hand_smoothed, right_hand_smoothed)
    cv2.imshow('Hand Tracking Overlay', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()