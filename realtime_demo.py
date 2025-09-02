import argparse
import platform
import re
import shutil
import sys
import subprocess
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple


def list_devices_with_ffmpeg(os_name: str) -> dict:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return {}
    try:
        if os_name == "Windows":
            cmd = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
        elif os_name == "Darwin":
            cmd = ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
        else:
            return {}
        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = (proc.stderr or "") + (proc.stdout or "")
        devices = {}
        if os_name == "Windows":
   
            for line in output.splitlines():
                if "DirectShow video devices" in line:
                    in_video_section = True
                    continue
                if in_video_section:
                    m = re.search(r"\[\s*(\d+)\s*\]\s+(.*)$", line)
                    if m:
                        try:
                            devices[int(m.group(1))] = m.group(2).strip().strip('"')
                        except Exception:
                            pass
                    if "DirectShow audio devices" in line:
                        break
        elif os_name == "Darwin":
            for line in output.splitlines():
                m = re.search(r"\[\s*(\d+)\s*\]\s+(.*)$", line)
                if m:
                    try:
                        devices[int(m.group(1))] = m.group(2).strip().strip('"')
                    except Exception:
                        pass
        return devices
    except Exception:
        return {}


def probe_camera_backend(index: int, os_name: str):
    backends = []
    if os_name == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    elif os_name == "Darwin":
        backends = [cv2.CAP_AVFOUNDATION]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.release()
                return backend
        cap.release()
    return None


def select_camera(prefer_name: Optional[str], index: Optional[int], max_tests: int = 10):
    os_name = platform.system()

    if index is not None:
        backend = probe_camera_backend(index, os_name)
        if backend is not None:
            return index, backend
        print(f"Provided camera index {index} is not available.")

    devices = list_devices_with_ffmpeg(os_name)
    if prefer_name and devices:
        for dev_index, dev_name in devices.items():
            if prefer_name.lower() in dev_name.lower():
                backend = probe_camera_backend(dev_index, os_name)
                if backend is not None:
                    return dev_index, backend

    for i in range(max_tests):
        backend = probe_camera_backend(i, os_name)
        if backend is not None:
            return i, backend



parser = argparse.ArgumentParser(description="Realtime Demo - AI Coach")
parser.add_argument("--index", type=int, default=None, help="Camera index to use (overrides auto-detection)")
parser.add_argument("--prefer-name", type=str, default="iriun", help="Preferred camera name substring to match (e.g., 'iriun')")
parser.add_argument("--max-tests", type=int, default=10, help="Maximum indices to probe when auto-detecting")
args = parser.parse_args()

selected_index, selected_backend = select_camera(
    prefer_name=args.prefer_name,
    index=args.index,
    max_tests=args.max_tests,
)

if selected_index is None or selected_backend is None:
    os_name = platform.system()
    devices = list_devices_with_ffmpeg(os_name)
    if devices:
        print("No working camera found. Devices detected by ffmpeg:")
        for i, name in devices.items():
            print(f"  [{i}] {name}")
        print("Try passing --index to select one of the above.")
    else:
        print("No working camera found. Try passing --index or ensure a camera is connected.")
    sys.exit(1)

print(f"✅ Using camera index {selected_index} with backend {selected_backend} on {platform.system()}")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

cap = cv2.VideoCapture(selected_index, selected_backend)

counter = 0
stage = None

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

            if hip.visibility > 0.7 and knee.visibility > 0.7 and ankle.visibility > 0.7:
                hip_coord = [hip.x, hip.y]
                knee_coord = [knee.x, knee.y]
                ankle_coord = [ankle.x, ankle.y]

                angle = calculate_angle(hip_coord, knee_coord, ankle_coord)

                cv2.putText(image, f'Knee Angle: {int(angle)} deg', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if angle > 160:
                    stage = "up"
                if angle < 90 and stage == 'up':
                    stage = "down"
                    counter += 1

                if angle >= 160:
                    quality = "Poor"
                    feedback = "Go Lower!"
                    color = (0, 0, 255)
                elif 90 <= angle < 160:
                    quality = "Good"
                    feedback = ""
                    color = (0, 255, 0)
                else:
                    quality = "Too Low"
                    feedback = "Don't go too deep!"
                    color = (0, 255, 255)

                cv2.putText(image, f'Reps: {counter}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f'Quality: {quality}', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                if feedback:
                    cv2.putText(image, feedback, (200, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
            else:
                cv2.putText(image, "Full body not in view - Step back",
                            (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        except Exception:
            pass

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        cv2.imshow('Realtime Demo (AI Coach)', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows() 