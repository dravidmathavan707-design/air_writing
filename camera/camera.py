from __future__ import annotations

import sys

import cv2


def _open_capture(index: int):
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    cap = cv2.VideoCapture(index)
    return cap if cap.isOpened() else None


class Camera:
    def __init__(self, index: int, width: int, height: int):
        self.width = width
        self.height = height
        self.index = index
        self.cap = _open_capture(index)
        if self.cap is None:
            raise RuntimeError("Could not open webcam.")
        self._apply_size()

    def _apply_size(self) -> None:
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def switch(self, index: int) -> bool:
        if index == self.index:
            return True
        new_cap = _open_capture(index)
        if new_cap is None:
            print(f"CAMERA {index} NOT AVAILABLE")
            return False
        self.cap.release()
        self.cap = new_cap
        self.index = index
        self._apply_size()
        print("CAMERA: LAPTOP" if index == 0 else "CAMERA: WEB")
        return True

    def read(self):
        success, frame = self.cap.read()
        if not success:
            return success, frame
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        if width != self.width or height != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return True, frame

    def release(self):
        self.cap.release()
