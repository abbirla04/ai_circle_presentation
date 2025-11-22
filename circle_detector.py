# circle_detector.py
import cv2, time, numpy as np
import mediapipe as mp
from collections import deque
mp_hands = mp.solutions.hands

class HandTracker:
    def __init__(self, max_len=64):
        self.hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
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
    # Fit circle (algebraic) to points; return center, radius, mean residual
    if len(points) < 6: return None
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = c
    r = np.sqrt(cx**2 + cy**2 + c0)
    residuals = np.sqrt((x - cx)**2 + (y - cy)**2)
    return (int(cx), int(cy)), float(r), float(np.mean(np.abs(residuals - r)))

# Quick demo loop
if __name__ == "__main__":
    ht = HandTracker()
    cap = cv2.VideoCapture(0)
    drawing = False
    last_detect_time = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame, pts = ht.process(frame)
        for p in pts: cv2.circle(frame, p, 3, (0,255,0), -1)
        circle = fit_circle(pts)
        if circle:
            (cx,cy), r, err = circle
            if err < 15 and r>30:
                cv2.circle(frame, (cx,cy), int(r), (0,0,255), 2)
                cv2.putText(frame, "CIRCLE!", (cx-40, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Demo", frame)
        k = cv2.waitKey(1) & 0xFF
        if k==27: break
    cap.release()
    cv2.destroyAllWindows()
