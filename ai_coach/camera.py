import platform
import re
import shutil
import subprocess
from typing import Dict, Optional, Tuple

import cv2


def list_devices_with_ffmpeg(os_name: str) -> Dict[int, str]:
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
        devices: Dict[int, str] = {}
        if os_name == "Windows":
            in_video_section = False
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


def probe_camera_backend(index: int, os_name: str) -> Optional[int]:
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


def select_camera(prefer_name: Optional[str], index: Optional[int], max_tests: int = 10) -> Tuple[Optional[int], Optional[int]]:
    os_name = platform.system()

    if index is not None:
        backend = probe_camera_backend(index, os_name)
        if backend is not None:
            return index, backend

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

    return None, None


def open_camera(prefer_name: Optional[str] = "iriun", index: Optional[int] = None, max_tests: int = 10):
    i, backend = select_camera(prefer_name=prefer_name, index=index, max_tests=max_tests)
    if i is None or backend is None:
        return None, None, list_devices_with_ffmpeg(platform.system())
    return cv2.VideoCapture(i, backend), (i, backend), {} 