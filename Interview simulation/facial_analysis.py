# facial_analysis.py
import cv2
import threading
import time
from collections import defaultdict
from deepface import DeepFace


class FacialExpressionAnalyzer:
    """
    Runs in background thread during interview.
    - Per question: start_question() / stop_question() → short summary
    - Full session: get_session_summary() → overall summary
    """

    def __init__(self, camera_index=0, analysis_interval=0.5):
        self.camera_index = camera_index
        self.analysis_interval = analysis_interval

        self._session_log = []        # all frames entire session
        self._question_log = []       # frames for current question only
        self._per_question = []       # finalized per-question summaries

        self.is_running = False
        self._recording = False
        self._thread = None
        self.cap = None

        self._latest_emotion = "Neutral"

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self):
        """Start background capture thread."""
        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop everything. Returns session summary."""
        self.is_running = False
        self._recording = False
        if self._thread:
            self._thread.join(timeout=3)
        if self.cap:
            self.cap.release()
        return self.get_session_summary()

    # ── Per-question control ──────────────────────────────────

    def start_question(self):
        """Call just before candidate starts answering."""
        self._question_log = []
        self._recording = True

    def stop_question(self, q_num: int) -> dict:
        """
        Call after candidate finishes answering.
        Returns short per-question summary.
        """
        self._recording = False
        summary = self._summarise(self._question_log)
        summary["question_num"] = q_num
        self._per_question.append(summary)
        return summary

    # ── Background capture loop ───────────────────────────────

    def _capture_loop(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            try:
                results = DeepFace.analyze(
                    img_path=frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                if results:
                    scores = results[0]['emotion']
                    dominant = results[0]['dominant_emotion']
                    self._latest_emotion = dominant
                    self._session_log.append(scores)
                    if self._recording:
                        self._question_log.append(scores)
            except Exception:
                pass
            time.sleep(self.analysis_interval)

    # ── Summaries ─────────────────────────────────────────────

    def _summarise(self, log: list) -> dict:
        """Average emotion scores from a list of frame results."""
        if not log:
            return {"dominant": "Neutral", "scores": {}}

        totals = defaultdict(float)
        for entry in log:
            for emotion, score in entry.items():
                totals[emotion] += score

        n = len(log)
        averages = {e: round(v / n, 2) for e, v in totals.items()}
        dominant = max(averages, key=averages.get)
        return {"dominant": dominant, "scores": averages}

    def get_session_summary(self) -> dict:
        """Full session summary + per-question breakdown."""
        overall = self._summarise(self._session_log)
        return {
            "overall_dominant": overall["dominant"],
            "overall_scores": overall["scores"],
            "per_question": self._per_question,
        }

    @property
    def latest_emotion(self) -> str:
        return self._latest_emotion