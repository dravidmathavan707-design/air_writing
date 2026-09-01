from __future__ import annotations

import cv2


class Camera:

    def __init__(
        self,
        index: int,
        width: int,
        height: int
    ):

        self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():

            raise RuntimeError(
                "Could not open webcam."
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            height
        )
        self.width = width
        self.height = height

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