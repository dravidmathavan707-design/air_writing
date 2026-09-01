from __future__ import annotations

from typing import Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np

from face.face_landmarks import assess_face_quality, bbox_from_points
from face.face_renderer import draw_face_portrait_from_points


def select_best_face_frame(samples):
    if not samples:
        return None

    return max(
        samples,
        key=lambda sample: (
            float(sample.get("confidence", 0.0))
            - float(sample.get("landmark_variance", 0.0)) * 0.05
        ),
    )


class FaceDetector:
    def __init__(
        self,
        detection_confidence=0.4,
        tracking_confidence=0.4,
    ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=8,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True
        return results

    def has_face(self, results):
        return bool(results and results.multi_face_landmarks)

    def _landmarks_to_points(self, landmarks, frame_shape):
        height, width = frame_shape[:2]
        points = []
        for landmark in landmarks.landmark:
            x = max(0, min(width - 1, int(round(landmark.x * width))))
            y = max(0, min(height - 1, int(round(landmark.y * height))))
            points.append((x, y))
        return points

    def landmarks_to_array(self, landmarks, frame_shape) -> np.ndarray:
        height, width = frame_shape[:2]
        return np.array(
            [[lm.x * width, lm.y * height] for lm in landmarks.landmark],
            dtype=np.float32,
        )

    def landmark_visibilities(self, landmarks) -> list:
        values = []
        for lm in landmarks.landmark:
            visibility = getattr(lm, "visibility", 0.0) or 0.0
            presence = getattr(lm, "presence", 0.0) or 0.0
            values.append(max(float(visibility), float(presence), 1.0 if visibility == 0 and presence == 0 else 0.0))
        return values

    def extract_faces(self, results, frame_shape):
        """Return every Face Mesh on screen, left-to-right."""
        faces = []
        if not self.has_face(results):
            return faces

        for landmarks in results.multi_face_landmarks:
            points = self.landmarks_to_array(landmarks, frame_shape)
            visibilities = self.landmark_visibilities(landmarks)
            quality = assess_face_quality(points, frame_shape, visibilities)
            faces.append(
                {
                    "points": points,
                    "visibilities": visibilities,
                    "quality": quality,
                    "bbox": bbox_from_points(points),
                }
            )

        faces.sort(
            key=lambda face: float(face["points"][1][0])
            if face["points"] is not None and len(face["points"]) > 1
            else 0.0
        )
        return faces

    def extract_face(self, results, frame_shape):
        faces = self.extract_faces(results, frame_shape)
        if not faces:
            return None, None, "not_detected"
        first = faces[0]
        return first["points"], first["visibilities"], first["quality"]

    def get_face_center(self, results, frame_shape):
        points, _, quality = self.extract_face(results, frame_shape)
        if points is None or quality == "not_detected" or len(points) < 5:
            return None
        return (int(points[1][0]), int(points[1][1]))

    def draw_face_diagram_from_points(
        self,
        canvas,
        points,
        color=(0, 255, 255),
        thickness=2,
        progress=1.0,
        hair_contour: Optional[np.ndarray] = None,
    ):
        if points is None or len(points) < 468:
            return False
        if progress < 1.0:
            count = max(8, int(len(points) * max(0.0, min(1.0, progress))))
            points = np.asarray(points)[:count]
            if len(points) < 468:
                return False
        draw_face_portrait_from_points(
            canvas,
            points,
            color=color,
            thickness=thickness,
            hair_contour=hair_contour,
        )
        return True

    def draw_face_diagram(
        self,
        canvas,
        results,
        frame_shape,
        color=(0, 255, 255),
        thickness=2,
        progress=1.0,
        hair_contour: Optional[Sequence] = None,
    ):
        points, _, quality = self.extract_face(results, frame_shape)
        if points is None or quality != "ok":
            return False
        return self.draw_face_diagram_from_points(
            canvas,
            points,
            color,
            thickness,
            progress,
            hair_contour=hair_contour,
        )

    def close(self):
        self.face_mesh.close()
