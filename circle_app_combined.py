import cv2
import numpy as np
import mediapipe as mp
import json
from collections import deque

# Load explanations
with open("explanations.json", "r") as f:
    explanations = json.load(f)

terms = list(explanations.keys())
current_term_index = 0

mp_hands = mp.solutions.hands

class HandTracker:
    def __init__(self, max_len=64):
        self.hands = mp_hands.Hands(min_detection_confidence=0.6,
                                    min_tracking_confidence=0.6)
        self.mp_draw = mp.solutions.drawing_utils
        self.points = deque(maxlen=max_len)

    def process(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(img_rgb)
        h, w = frame.shape[:2]
        tip = None

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lm = hand.landmark[8]  # index finger tip
            tip = (int(lm.x * w), int(lm.y * h))
            self.mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            self.points.append(tip)

        return frame, list(self.points)

def fit_circle(points):
    if len(points) < 6:
        return None
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = c
    r = np.sqrt(cx**2 + cy**2 + c0)
    residuals = np.sqrt((x - cx)**2 + (y - cy)**2)
    return (int(cx), int(cy)), float(r), float(np.mean(np.abs(residuals - r)))


# --------------------------------------------------------
#                     MAIN LOOP
# --------------------------------------------------------
if __name__ == "__main__":

    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    popup_active = False
    popup_timer = 0
    popup_duration = 4  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, pts = tracker.process(frame)

        # Draw tracked points
        for p in pts:
            cv2.circle(frame, p, 3, (0, 255, 0), -1)

        circle = fit_circle(pts)
        if circle:
            (cx, cy), r, err = circle
            if err < 15 and r > 30 and not popup_active:
                popup_active = True
                popup_timer = cv2.getTickCount()

        # -------------------------------
        # Popup overlay logic
        # -------------------------------
        if popup_active:
            elapsed = (cv2.getTickCount() - popup_timer) / cv2.getTickFrequency()
            if elapsed < popup_duration:
                term = terms[current_term_index]
                explanation = explanations[term]

                # Draw semi-transparent popup
                overlay = frame.copy()
                cv2.rectangle(overlay, (50, 50), (600, 250), (0, 0, 0), -1)
                alpha = 0.55
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                cv2.putText(frame, f"Topic: {term}", (70, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.putText(frame, explanation, (70, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            else:
                popup_active = False
                current_term_index = (current_term_index + 1) % len(terms)

        cv2.imshow("AI Circle Presentation", frame)
        key = cv2.waitKey(1)

        if key == ord('n'):
            current_term_index = (current_term_index + 1) % len(terms)
        if key == ord('p'):
            current_term_index = (current_term_index - 1) % len(terms)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
