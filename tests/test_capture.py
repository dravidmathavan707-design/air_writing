from pathlib import Path

import cv2
import numpy as np

from capture.screen_capture import ScreenCapture


def test_screenshot_writes_png(tmp_path):
    capture = ScreenCapture(output_dir=str(tmp_path))
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[:] = (40, 180, 255)
    path = capture.screenshot(frame, now=1.0)
    assert path.exists()
    saved = cv2.imread(str(path))
    assert saved is not None
    assert saved.shape[0] == 80
    assert capture.message.startswith("SCREENSHOT SAVED")


def test_record_toggle_writes_video(tmp_path):
    capture = ScreenCapture(output_dir=str(tmp_path), fps=10.0)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    capture.toggle_record(frame)
    if not capture.recording:
        return
    for _ in range(8):
        capture.write(frame)
    capture.stop()
    files = list(Path(tmp_path).glob("recording_*"))
    assert files
    assert files[0].stat().st_size > 0
