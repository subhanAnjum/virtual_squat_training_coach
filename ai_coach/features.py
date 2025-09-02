from typing import Dict, List

import numpy as np
from .pose import compute_basic_angles
from .exercises import SquatAdvisor


FEATURE_ORDER = [
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_elbow",
    "right_elbow",
    "torso_lean_deg",
    "knee_valgus_ratio",
]


def extract_feature_vector(landmarks) -> np.ndarray:
    angles: Dict[str, float] = compute_basic_angles(landmarks)
    torso_lean = SquatAdvisor.torso_lean_degrees(landmarks)
    knee_valgus = SquatAdvisor.knee_valgus_ratio(landmarks) or 1.0
    vector = [
        angles.get("left_knee", 0.0),
        angles.get("right_knee", 0.0),
        angles.get("left_hip", 0.0),
        angles.get("right_hip", 0.0),
        angles.get("left_elbow", 0.0),
        angles.get("right_elbow", 0.0),
        float(torso_lean),
        float(knee_valgus),
    ]
    return np.array(vector, dtype=np.float32) 