"""Save screenshots and record the on-screen window."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class ScreenCapture:
    def __init__(self, output_dir: Optional[str] = None, fps: float = 20.0):
        self.output_dir = Path(output_dir) if output_dir else Path.home() / "Videos"
        self.fps = fps
        self.writer: Optional[cv2.VideoWriter] = None
        self.recording = False
        self.video_path: Optional[Path] = None
        self.message = ""
        self.flash_until = 0.0
        self._size: Optional[Tuple[int, int]] = None

    def screenshot(self, frame: np.ndarray, now: float) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"screenshot_{_stamp()}.png"
        cv2.imwrite(str(path), frame)
        self.message = f"SCREENSHOT SAVED: {path.name}"
        self.flash_until = now + 0.35
        print(self.message)
        return path

    def toggle_record(self, frame: np.ndarray) -> None:
        if self.recording:
            self.stop()
            return
        self.start(frame)

    def start(self, frame: np.ndarray) -> None:
        self.stop()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        self._size = (width, height)
        stamp = _stamp()
        path, writer = _open_writer(self.output_dir, stamp, self._size, self.fps)
        if writer is None or not writer.isOpened():
            self.message = "RECORD FAILED - CODEC NOT AVAILABLE"
            print(self.message)
            return
        self.writer = writer
        self.video_path = path
        self.recording = True
        self.message = f"RECORDING: {path.name}"
        print(self.message)

    def write(self, frame: np.ndarray) -> None:
        if not self.recording or self.writer is None:
            return
        height, width = frame.shape[:2]
        if self._size != (width, height):
            frame = cv2.resize(frame, self._size)
        self.writer.write(frame)

    def stop(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            if self.video_path is not None:
                self.message = f"VIDEO SAVED: {self.video_path.name}"
                print(self.message)
        self.recording = False
        self.video_path = None

    def close(self) -> None:
        self.stop()


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _open_writer(folder: Path, stamp: str, size: Tuple[int, int], fps: float):
    for fourcc_name, ext in (("mp4v", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")):
        path = folder / f"recording_{stamp}{ext}"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), fps, size)
        if writer.isOpened():
            return path, writer
        writer.release()
    return None, None
