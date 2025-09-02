from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np

from .exercises import ExerciseHeuristics
from .per_rep_features import SQUAT_REP_FEATURES


class ExerciseClassifier:
    def __init__(self, model_path: str = "models/exercise_classifier.joblib"):
        self.model_path = Path(model_path)
        self._model = None
        if self.model_path.exists():
            try:
                self._model = joblib.load(self.model_path)
            except Exception:
                self._model = None

    def predict_label(self, feature_vector: np.ndarray, angles: Dict[str, float]) -> str:
        if self._model is not None:
            try:
                pred = self._model.predict(feature_vector.reshape(1, -1))
                return str(pred[0])
            except Exception:
                pass
        return ExerciseHeuristics.classify(angles)


class SquatQualityClassifier:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = "models/squat_quality_multiclass.joblib"
        
        self.model_path = Path(model_path)
        self._model = None
        if self.model_path.exists():
            try:
                self._model = joblib.load(self.model_path)
            except Exception:
                self._model = None

    def is_available(self) -> bool:
        return self._model is not None

    def predict_quality(self, per_rep_vector: np.ndarray) -> Optional[str]:
        if self._model is None:
            return None
        try:
            pred = self._model.predict(per_rep_vector.reshape(1, -1))
            return str(pred[0])
        except Exception:
            return None 