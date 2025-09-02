from dataclasses import dataclass, field
from typing import Dict, Optional
import time
import numpy as np

from .pose import compute_basic_angles
from .exercises import SquatAdvisor

SQUAT_REP_FEATURES = [
    
    "min_left_knee", "min_right_knee", "max_left_knee", "max_right_knee",
    "rom_left", "rom_right", "avg_torso_lean", "max_torso_lean",
    "std_torso_lean", "avg_knee_angle", "std_knee_angle", "min_knee_valgus",
    "duration_secs",
    "knee_angle_diff", "hip_angle_diff", "torso_lean_at_bottom",
    "descent_duration_ratio", "bottom_stability_lean"
]

@dataclass
class RepAggregate:
    start_time: float = field(default_factory=time.time)
    
    min_left_knee: float = 999.0
    min_right_knee: float = 999.0
    min_left_hip: float = 999.0
    min_right_hip: float = 999.0
    max_left_knee: float = 0.0
    max_right_knee: float = 0.0
    max_torso_lean: float = 0.0
    min_knee_valgus: float = 999.0

    
    frame_count: int = 0
    angle_series: list = field(default_factory=list)
    torso_lean_series: list = field(default_factory=list)

    def update(self, angles: Dict[str, float], landmarks) -> None:
        self.frame_count += 1
        
        self.angle_series.append(angles)

        
        lk, rk = float(angles.get("left_knee", 180.0)), float(angles.get("right_knee", 180.0))
        lh, rh = float(angles.get("left_hip", 180.0)), float(angles.get("right_hip", 180.0))
        
        self.min_left_knee = min(self.min_left_knee, lk)
        self.min_right_knee = min(self.min_right_knee, rk)
        self.min_left_hip = min(self.min_left_hip, lh)
        self.min_right_hip = min(self.min_right_hip, rh)
        
        self.max_left_knee = max(self.max_left_knee, lk)
        self.max_right_knee = max(self.max_right_knee, rk)

        torso_lean = float(SquatAdvisor.torso_lean_degrees(landmarks))
        self.torso_lean_series.append(torso_lean)
        self.max_torso_lean = max(self.max_torso_lean, torso_lean)

        kv = SquatAdvisor.knee_valgus_ratio(landmarks)
        if kv is not None:
            self.min_knee_valgus = min(self.min_knee_valgus, float(kv))

    def to_feature_vector(self) -> Optional[np.ndarray]:
        if not self.angle_series:
            return None

        
        knee_angles = np.array([min(f.get('left_knee', 180), f.get('right_knee', 180)) for f in self.angle_series])
        bottom_frame_idx = np.argmin(knee_angles)

        
        duration = max(time.time() - self.start_time, 1e-3)
        rom_left = max(0.0, self.max_left_knee - self.min_left_knee)
        rom_right = max(0.0, self.max_right_knee - self.min_right_knee)
        min_kv = self.min_knee_valgus if self.min_knee_valgus != 999.0 else 1.0
        avg_torso_lean = float(np.mean(self.torso_lean_series)) if self.torso_lean_series else 0.0
        std_torso_lean = float(np.std(self.torso_lean_series)) if self.torso_lean_series else 0.0
        avg_knee_angle = float(np.mean(knee_angles)) if knee_angles.size > 0 else 0.0
        std_knee_angle = float(np.std(knee_angles)) if knee_angles.size > 0 else 0.0

        
        knee_angle_diff = abs(self.min_left_knee - self.min_right_knee)
        hip_angle_diff = abs(self.min_left_hip - self.min_right_hip)        
        
        torso_lean_at_bottom = self.torso_lean_series[bottom_frame_idx] if self.torso_lean_series else 0.0
        descent_duration_ratio = (bottom_frame_idx + 1) / max(self.frame_count, 1)

        
        bottom_window_size = max(1, self.frame_count // 5) # 20% of frames around the bottom
        start_idx = max(0, bottom_frame_idx - bottom_window_size // 2)
        end_idx = min(self.frame_count, bottom_frame_idx + bottom_window_size // 2 + 1)
        bottom_lean_series = self.torso_lean_series[start_idx:end_idx]
        bottom_stability_lean = float(np.std(bottom_lean_series)) if bottom_lean_series else 0.0

        vec = np.array([
            self.min_left_knee, self.min_right_knee, self.max_left_knee, self.max_right_knee,
            rom_left, rom_right, avg_torso_lean, self.max_torso_lean,
            std_torso_lean, avg_knee_angle, std_knee_angle, min_kv,
            duration,
            knee_angle_diff, hip_angle_diff, torso_lean_at_bottom,
            descent_duration_ratio, bottom_stability_lean
        ], dtype=np.float32)
        
        return vec
