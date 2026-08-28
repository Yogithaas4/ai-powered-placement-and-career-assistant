# camera.py
import cv2
import threading
import time
import numpy as np
from facial_analysis import FacialExpressionAnalyzer
from config import CAMERA_INDEX, FRAME_INTERVAL

_EMOTION_COLORS = {
    "happy":    (0, 220, 80),
    "surprise": (0, 200, 255),
    "neutral":  (180, 180, 180),
    "sad":      (200, 80,  80),
    "fear":     (180, 0,   200),
    "disgust":  (0,  140, 180),
    "angry":    (0,  60,  220),
}


class CameraSession:
    """
    Handles webcam display in background thread.
    Delegates all emotion analysis to FacialExpressionAnalyzer.
    """

    def __init__(self):
        self._analyzer = FacialExpressionAnalyzer(
            camera_index=CAMERA_INDEX,
            analysis_interval=FRAME_INTERVAL / 30.0  # convert frames to seconds
        )
        self._cap = None
        self._thread = None
        self._stop_flag = threading.Event()
        self._recording = False
        self._current_q = 0

    def open(self):
        self._analyzer.start()
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()
        print("[Camera] Started.")

    def close(self):
        self._stop_flag.set()
        self._analyzer.stop()
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        print("[Camera] Closed.")

    def start_recording(self, q_num: int = 0):
        self._current_q = q_num
        self._recording = True
        self._analyzer.start_question()

    def stop_recording(self) -> dict:
        self._recording = False
        return self._analyzer.stop_question(self._current_q)

    def get_session_summary(self) -> dict:
        return self._analyzer.get_session_summary()

    def _display_loop(self):
        while not self._stop_flag.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            emotion = self._analyzer.latest_emotion
            color = _EMOTION_COLORS.get(emotion.lower(), (255, 255, 255))
            h, w = frame.shape[:2]

            # Emotion badge
            cv2.rectangle(frame, (10, 10), (270, 50), (0, 0, 0), -1)
            cv2.putText(frame, emotion.capitalize(), (15, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Question number
            if self._current_q > 0:
                cv2.putText(frame, f"Q{self._current_q}", (15, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # REC indicator
            if self._recording:
                cv2.circle(frame, (w - 25, 25), 10, (0, 0, 220), -1)
                cv2.putText(frame, "REC", (w - 80, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)

            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)
            cv2.imshow("Mock Interview", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._stop_flag.set()
                break