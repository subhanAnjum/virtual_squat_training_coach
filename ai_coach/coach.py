from typing import Dict, List, Optional, Tuple

import cv2

from .pose import compute_basic_angles
from .exercises import SquatAdvisor, SimpleRepCounter
from .per_rep_features import RepAggregate
from .classifier import SquatQualityClassifier


class HybridSquatCoach:
    def __init__(self, model_path: Optional[str] = None):
        self.counter = SimpleRepCounter()
        self.current_rep: Optional[RepAggregate] = None
        self.quality_model = SquatQualityClassifier(model_path=model_path)

        self.last_rep_quality: str = ""
        self.last_rep_recs: List[str] = []
        self.last_rep_color: Tuple[int, int, int] = (255, 255, 255)
        self.last_rep_index: int = 0

    def reset(self):
        self.counter = SimpleRepCounter()
        self.current_rep = None
        self.last_rep_quality = ""
        self.last_rep_recs = []
        self.last_rep_color = (255, 255, 255)
        self.last_rep_index = 0

    def update(self, landmarks) -> Dict:
        angles: Dict[str, float] = compute_basic_angles(landmarks)

        if self.current_rep is None:
            self.current_rep = RepAggregate()
        self.current_rep.update(angles, landmarks)

        # Frame-level rule hints
        _, frame_recs, frame_color = SquatAdvisor.quality_and_recommendations(angles, landmarks)

        rep_count_before = self.counter.reps
        rep_count_after = self.counter.update(angles)

        if rep_count_after > rep_count_before:
            ml_quality = None
            if self.quality_model.is_available():
                vec = self.current_rep.to_feature_vector()
                pred = self.quality_model.predict_quality(vec)
                ml_quality = pred.lower() if pred is not None else None

            if ml_quality is None:
                # Fallback to rule-based if ML model fails
                self.last_rep_quality = "Good" if not frame_recs else "Needs Work"
                self.last_rep_color = (0, 255, 0) if self.last_rep_quality == "Good" else (0, 165, 255)
            else:
                self.last_rep_quality = ml_quality.replace('_', ' ').title()
                if ml_quality == "good":
                    self.last_rep_color = (0, 255, 0)  # Green
                elif ml_quality == "bad_posture":
                    self.last_rep_color = (0, 165, 255)  # Orange
                else: # too_high, too_low
                    self.last_rep_color = (0, 0, 255)  # Red

            # Use rule-based recommendations to give specific hints
            self.last_rep_recs = SquatAdvisor.quality_and_recommendations(angles, landmarks)[1]
            self.last_rep_index = rep_count_after

            self.current_rep = RepAggregate()

        if self.last_rep_index > 0:
            display_quality = f"Quality (Rep {self.last_rep_index}): {self.last_rep_quality}"
            display_recs = [f"Feedback (Rep {self.last_rep_index}): {r}" for r in self.last_rep_recs]
            display_color = self.last_rep_color
        else:
            display_quality = "Quality: Good" if not frame_recs else "Quality: Needs Work"
            display_recs = frame_recs
            display_color = frame_color

        return {
            "rep_count": self.counter.reps,
            "display_quality": display_quality,
            "display_recs": display_recs,
            "display_color": display_color,
        }
