from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose


@dataclass
class SimpleRepCounter:
    """Rep counter using fixed knee-angle thresholds.
    Uses left knee angle from precomputed angles dict.
    """
    UP_THRESHOLD = 150.0  # More lenient 'up' state
    DOWN_THRESHOLD = 105.0 # Must go at least this low to count
    stage: Optional[str] = None
    reps: int = 0
    last_rep_rom: Optional[float] = None
    current_min: Optional[float] = None
    current_max: Optional[float] = None

    def update(self, angles: Dict[str, float]) -> int:
        knee_angle = angles.get("left_knee")
        if knee_angle is None:
            return self.reps

        # Track ROM for info
        self.current_min = knee_angle if self.current_min is None else min(self.current_min, knee_angle)
        self.current_max = knee_angle if self.current_max is None else max(self.current_max, knee_angle)

        if knee_angle > self.UP_THRESHOLD:
            self.stage = "up"
        if knee_angle < self.DOWN_THRESHOLD and self.stage == "up":
            self.stage = "down"
            self.reps += 1
            if self.current_min is not None and self.current_max is not None:
                self.last_rep_rom = abs(self.current_max - self.current_min)
            # reset for next cycle
            self.current_min = None
            self.current_max = None

        return self.reps


class SquatAdvisor:
    # --- Constants based on biomechanics research ---
  
    GOOD_KNEE_ANGLE_MIN = 30.0  
    GOOD_KNEE_ANGLE_MAX = 140.0

    MAX_TORSO_LEAN_DEGREES = 55.0

    MIN_ANKLE_ANGLE = 70.0

    MIN_KNEE_VALGUS_RATIO = 0.85

    @staticmethod
    def squat_depth_angle(angles: Dict[str, float]) -> float:
        left_knee = angles.get("left_knee", 180.0)
        right_knee = angles.get("right_knee", 180.0)
        return float(min(left_knee, right_knee))

    @staticmethod
    def torso_lean_degrees(landmarks) -> float:
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        def angle_to_vertical(hip, sh):
            vx = sh.x - hip.x
            vy = sh.y - hip.y
            norm = max(np.hypot(vx, vy), 1e-6)
            cos_theta = (-vy) / norm 
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            return float(np.degrees(np.arccos(cos_theta)))

        left = angle_to_vertical(l_hip, l_sh)
        right = angle_to_vertical(r_hip, r_sh)
        return float((left + right) / 2.0)

    @staticmethod
    def shin_angle_degrees(landmarks) -> float:
        """Computes the shin angle relative to the vertical, averaged for both legs."""
        l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        def angle_to_vertical(joint1, joint2): 
            vx = joint2.x - joint1.x
            vy = joint2.y - joint1.y
            norm = max(np.hypot(vx, vy), 1e-6)
            cos_theta = (-vy) / norm
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            return float(np.degrees(np.arccos(cos_theta)))

        left = angle_to_vertical(l_ankle, l_knee)
        right = angle_to_vertical(r_ankle, r_knee)
        return float((left + right) / 2.0)

    @staticmethod
    def ankle_angle_degrees(landmarks) -> float:
        """Computes the ankle angle, averaged for both legs."""
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        l_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        r_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]

        def calculate_angle(p1, p2, p3):
            v1 = np.array(p1) - np.array(p2)
            v2 = np.array(p3) - np.array(p2)
            angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
            return angle

        left = calculate_angle(l_heel, l_ankle, l_knee)
        right = calculate_angle(r_heel, r_ankle, r_knee)
        return float((left + right) / 2.0)

    @staticmethod
    def knee_valgus_ratio(landmarks) -> Optional[float]:
        lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        ankle_sep = abs(la.x - ra.x)
        knee_sep = abs(lk.x - rk.x)
        if ankle_sep < 1e-3:
            return None
        return float(knee_sep / ankle_sep)

    @staticmethod
    def quality_and_recommendations(angles: Dict[str, float], landmarks) -> Tuple[str, List[str], Tuple[int, int, int]]:
        depth = SquatAdvisor.squat_depth_angle(angles)
        recs: List[str] = []
        quality = "Good" 
        color = (0, 255, 0)

        

        
        if depth > SquatAdvisor.GOOD_KNEE_ANGLE_MAX:
            quality = "Too High"
            color = (0, 165, 255) # Orange
            recs.append(f"Go deeper. Aim for thighs parallel to the floor.")
        elif depth < SquatAdvisor.GOOD_KNEE_ANGLE_MIN:
            quality = "Too Low"
            color = (0, 0, 255) # Red
            recs.append("Avoid excessive depth unless you have the mobility for it.")
        
        
        
        
        try:
        
            torso_lean = SquatAdvisor.torso_lean_degrees(landmarks)
            ankle_angle = SquatAdvisor.ankle_angle_degrees(landmarks)
            
            if torso_lean > SquatAdvisor.MAX_TORSO_LEAN_DEGREES and ankle_angle < SquatAdvisor.MIN_ANKLE_ANGLE:
                quality = "Bad Posture"
                color = (0, 165, 255) 
                recs.append("Keep your chest up. Brace your core.")

            
            ratio = SquatAdvisor.knee_valgus_ratio(landmarks)
            if ratio is not None and ratio < SquatAdvisor.MIN_KNEE_VALGUS_RATIO:
                quality = "Bad Posture"
                color = (0, 165, 255) 
                recs.append("Push your knees out over your feet.")
        except Exception:
            pass

        if quality == "Bad Posture" and not any("chest up" in r or "knees out" in r for r in recs):
             pass
        elif quality != "Good" and len(recs) == 0:
             if quality == "Too High": recs.append(f"Go deeper.")
             if quality == "Too Low": recs.append(f"Control your depth.")


        if len(recs) > 2:
            recs = recs[:2]
        return quality, recs, color


class ExerciseHeuristics:
    @staticmethod
    def classify(angles: Dict[str, float]) -> str:
        left_knee = angles.get("left_knee", 180)
        right_knee = angles.get("right_knee", 180)
        left_elbow = angles.get("left_elbow", 180)
        right_elbow = angles.get("right_elbow", 180)

        if left_knee < 120 or right_knee < 120:
            return "squat"
        if left_elbow < 120 or right_elbow < 120:
            return "bicep_curl"
        return "unknown"

    @staticmethod
    def quality_feedback(label: str, angles: Dict[str, float]) -> tuple[str, str, tuple]:
        if label == "squat":
            knee = min(angles.get("left_knee", 180), angles.get("right_knee", 180))
            if knee >= 160:
                return "Poor", "Go Lower!", (0, 0, 255)
            if 90 <= knee < 160:
                return "Good", "", (0, 255, 0)
            return "Too Low", "Don't go too deep!", (0, 255, 255)
        if label == "bicep_curl":
            elbow = min(angles.get("left_elbow", 180), angles.get("right_elbow", 180))
            if elbow < 60:
                return "Good", "", (0, 255, 0)
            if elbow < 120:
                return "OK", "Squeeze at top", (0, 255, 255)
            return "Poor", "Curl more", (0, 0, 255)
        return "N/A", "", (255, 255, 255)


class BicepCurlAdvisor:
    @staticmethod
    def quality_and_recommendations(angles: Dict[str, float], landmarks) -> Tuple[str, List[str], Tuple[int, int, int]]:
        left = angles.get("left_elbow", 180.0)
        right = angles.get("right_elbow", 180.0)
        elbow = min(left, right)
        recs: List[str] = []
        if elbow < 60:
            quality = "Good"
            color = (0, 255, 0)
        elif elbow < 120:
            quality = "OK"
            color = (0, 255, 255)
            recs.append("Curl higher: flex your elbow until your forearm is close to your biceps")
        else:
            quality = "Poor"
            color = (0, 0, 255)
            recs.append("Keep your upper arm still and curl by bending at the elbow; avoid swinging the torso")
        if len(recs) > 2:
            recs = recs[:2]
        return quality, recs, color 