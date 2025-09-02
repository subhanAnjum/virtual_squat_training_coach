from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle_3d(point_a: Tuple[float, float, float],
                       point_b: Tuple[float, float, float],
                       point_c: Tuple[float, float, float]) -> float:
    a = np.array(point_a, dtype=float)
    b = np.array(point_b, dtype=float)
    c = np.array(point_c, dtype=float)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return 0.0
    cos_theta = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos_theta)))
    # Normalize to [0,180]
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def get_landmark_tuple(landmarks, landmark_enum) -> Tuple[float, float, float, float]:
    lm = landmarks[landmark_enum.value]
    return float(lm.x), float(lm.y), float(lm.z), float(lm.visibility)


def compute_basic_angles(landmarks) -> Dict[str, float]:
    angles: Dict[str, float] = {}
    # Left side
    lx, ly, lz, lv = get_landmark_tuple(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    lkx, lky, lkz, lkv = get_landmark_tuple(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
    lax, lay, laz, lav = get_landmark_tuple(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
    lsx, lsy, lsz, lsv = get_landmark_tuple(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    lex, ley, lez, lev = get_landmark_tuple(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
    lwmx, lwmy, lwmz, lwv = get_landmark_tuple(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)

    # Right side
    rx, ry, rz, rv = get_landmark_tuple(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
    rkx, rky, rkz, rkv = get_landmark_tuple(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
    rax, ray, raz, rav = get_landmark_tuple(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
    rsx, rsy, rsz, rsv = get_landmark_tuple(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    rex, rey, rez, rev = get_landmark_tuple(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
    rwmx, rwmy, rwmz, rwv = get_landmark_tuple(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)

    # Compute 3D angles
    angles["left_knee"] = calculate_angle_3d((lx, ly, lz), (lkx, lky, lkz), (lax, lay, laz))
    angles["right_knee"] = calculate_angle_3d((rx, ry, rz), (rkx, rky, rkz), (rax, ray, raz))

    angles["left_elbow"] = calculate_angle_3d((lsx, lsy, lsz), (lex, ley, lez), (lwmx, lwmy, lwmz))
    angles["right_elbow"] = calculate_angle_3d((rsx, rsy, rsz), (rex, rey, rez), (rwmx, rwmy, rwmz))

    angles["left_hip"] = calculate_angle_3d((lsx, lsy, lsz), (lx, ly, lz), (lkx, lky, lkz))
    angles["right_hip"] = calculate_angle_3d((rsx, rsy, rsz), (rx, ry, rz), (rkx, rky, rkz))

    return angles


class PoseEstimator:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._pose = mp_pose.Pose(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def close(self):
        """Releases the MediaPipe pose model."""
        if self._pose:
            self._pose.close()
            self._pose = None

    def process(self, frame_bgr) -> Tuple[Optional[object], object]:
        if not self._pose:
            raise RuntimeError("PoseEstimator has been closed and cannot process frames.")

        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self._pose.process(image)
        image.flags.writeable = True
        annotated = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return results, annotated

    def estimate_landmarks(self, frame_bgr) -> Optional[List]:
        """Processes a frame and returns only the pose world landmarks."""
        results, _ = self.process(frame_bgr)
        if results and results.pose_world_landmarks:
            return results.pose_world_landmarks.landmark
        return None

    @staticmethod
    def draw_landmarks(frame_bgr, pose_landmarks):
        if pose_landmarks is None:
            return
        mp_drawing.draw_landmarks(
            frame_bgr,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        ) 