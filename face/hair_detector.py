"""
Hair silhouette extraction near a detected face.

Uses a landmark-guided dark-region mask (no extra model required).
If a MediaPipe HairSegmenter .tflite is present, that path is used first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def draw_hair_contour(
    canvas: np.ndarray,
    contour: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> None:
    if contour is not None and len(contour) >= 3:
        cv2.drawContours(
            canvas,
            [contour.astype(np.int32)],
            -1,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )


class HairDetector:
    """Extract hair silhouette from the region around a face."""

    def __init__(
        self,
        threshold: int = 30,
        min_contour_area: int = 500,
        approx_epsilon_ratio: float = 0.004,
        morph_kernel_size: int = 5,
        smooth_iterations: int = 2,
        model_path: Optional[str] = None,
    ):
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.approx_epsilon_ratio = approx_epsilon_ratio
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self.smooth_iterations = smooth_iterations
        self.segmenter = None

        if model_path is None:
            model_path = str(
                Path(__file__).resolve().parent.parent / "models" / "hair_segmenter.tflite"
            )
        if Path(model_path).is_file():
            self._try_load_segmenter(str(Path(model_path).resolve()))

    def _try_load_segmenter(self, model_path: str) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            options = vision.ImageSegmenterOptions(
                base_options=python.BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True,
                output_confidence_masks=False,
            )
            self.segmenter = vision.ImageSegmenter.create_from_options(options)
            self._mp = mp
        except Exception as exc:
            print(f"Hair segmenter unavailable, using classical mask: {exc}")
            self.segmenter = None

    def detect_hair_mask(
        self,
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None

        if self.segmenter is not None:
            mask = self._segmenter_mask(frame)
        else:
            mask = self._classical_mask(frame, face_bbox)

        if mask is None:
            return None
        return self._restrict_to_face(mask, face_bbox)

    def _segmenter_mask(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.segmenter.segment(mp_image)
        category_mask = result.category_mask.numpy_view()
        hair_mask = (category_mask == 1).astype(np.uint8) * 255
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, self.morph_kernel)
        return hair_mask

    def _classical_mask(
        self,
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)

        if face_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in face_bbox]
            y1 = max(0, y1)
            y2 = min(gray.shape[0], y2)
            x1 = max(0, x1)
            x2 = min(gray.shape[1], x2)
            cheek = blur[int(y1 + 0.35 * (y2 - y1)) : int(y1 + 0.7 * (y2 - y1)), x1:x2]
            if cheek.size:
                skin = float(np.median(cheek))
                threshold = max(self.threshold, int(skin * 0.72))
            else:
                threshold = self.threshold
        else:
            threshold = self.threshold

        _, mask = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        return mask

    def _restrict_to_face(
        self,
        mask: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        if face_bbox is None:
            return mask

        x1, y1, x2, y2 = face_bbox
        h, w = mask.shape
        face_h = max(1, y2 - y1)
        face_w = max(1, x2 - x1)
        pad_y = int(face_h * 0.85)
        pad_x = int(face_w * 0.28)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        # Hair lives above the eyes, not on the beard / jaw.
        y2 = min(h, int(y1 + pad_y + face_h * 0.38))

        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        return roi_mask

    def get_hair_contour(
        self,
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[np.ndarray]:
        mask = self.detect_hair_mask(frame, face_bbox)
        if mask is None:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest = None
        for cnt in contours:
            if cv2.contourArea(cnt) >= self.min_contour_area:
                largest = cnt
                break
        if largest is None:
            return None

        peri = cv2.arcLength(largest, closed=True)
        epsilon = self.approx_epsilon_ratio * peri
        approx = cv2.approxPolyDP(largest, epsilon, closed=True)
        if len(approx) < 3:
            return None

        if self.smooth_iterations > 0:
            approx = self._smooth_contour(approx, iterations=self.smooth_iterations)
        return approx

    def extract_contour(
        self,
        frame_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[np.ndarray]:
        return self.get_hair_contour(frame_bgr, face_bbox=face_bbox)

    def _smooth_contour(self, contour: np.ndarray, iterations: int = 1) -> np.ndarray:
        pts = contour.reshape(-1, 2).astype(np.float32)
        for _ in range(iterations):
            new_pts = []
            n = len(pts)
            for i in range(n):
                p0 = pts[i]
                p1 = pts[(i + 1) % n]
                new_pts.append(0.75 * p0 + 0.25 * p1)
                new_pts.append(0.25 * p0 + 0.75 * p1)
            pts = np.array(new_pts, dtype=np.float32)
        return pts.reshape(-1, 1, 2).astype(np.int32)

    def close(self):
        if self.segmenter is not None:
            try:
                self.segmenter.close()
            except Exception:
                pass
