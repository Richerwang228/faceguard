import cv2
import numpy as np
from typing import List


def _compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute Intersection over Union of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def _nms(boxes: List[List[int]], scores: List[float], threshold: float = 0.3) -> List[int]:
    """Apply Non-Maximum Suppression to remove overlapping boxes."""
    if not boxes:
        return []

    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []

    while indices:
        current = indices[0]
        keep.append(current)
        current_box = boxes[current]

        remaining = []
        for idx in indices[1:]:
            iou = _compute_iou(current_box, boxes[idx])
            if iou < threshold:
                remaining.append(idx)
        indices = remaining

    return keep


class FaceDetector:
    """Face detector using OpenCV Haar Cascade with profile face fallback."""

    def __init__(self, min_detection_confidence: float = 0.5):
        self.min_confidence = min_detection_confidence

        # Frontal face cascade (primary)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        self.frontal_cascade = cv2.CascadeClassifier(cascade_path)
        if self.frontal_cascade.empty():
            self.frontal_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

        # Profile face cascade (fallback for side faces)
        profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        self.profile_cascade = cv2.CascadeClassifier(profile_path)

    def _detect_single_cascade(
        self,
        cascade: cv2.CascadeClassifier,
        gray: np.ndarray,
        image_shape: tuple,
        scale_factor: float = 1.1,
        min_neighbors: int = 3,
    ) -> List[dict]:
        """Run a single cascade detector and return normalized results."""
        h, w = image_shape[:2]
        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(48, 48),
            maxSize=(w, h),
        )

        faces = []
        if len(detections) == 0:
            return faces

        for (x, y, fw, fh) in detections:
            face_area = fw * fh
            img_area = w * h
            relative_size = face_area / img_area

            aspect_ratio = fw / fh if fh > 0 else 1.0
            aspect_score = 1.0 - abs(aspect_ratio - 1.0)
            score = min(1.0, 0.5 + relative_size * 5 + aspect_score * 0.3)

            if score >= self.min_confidence:
                faces.append({
                    'bbox': [int(x), int(y), int(x + fw), int(y + fh)],
                    'score': float(score),
                    'relative_bbox': [
                        float(x / w),
                        float(y / h),
                        float(fw / w),
                        float(fh / h)
                    ]
                })
        return faces

    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in a BGR image using frontal + profile cascades.
        Returns list of dicts with keys: bbox [x1,y1,x2,y2], score
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]

        # 1. Frontal detection (primary)
        frontal_faces = self._detect_single_cascade(
            self.frontal_cascade, gray, (h, w),
            scale_factor=1.1, min_neighbors=3,
        )

        all_faces = list(frontal_faces)

        # 2. Profile detection (fallback) - only if frontal finds few faces
        if self.profile_cascade is not None and not self.profile_cascade.empty():
            # Use slightly relaxed params for profile to catch more side faces
            profile_faces = self._detect_single_cascade(
                self.profile_cascade, gray, (h, w),
                scale_factor=1.15, min_neighbors=2,
            )

            # Filter out profile detections that heavily overlap with frontal
            for pf in profile_faces:
                is_duplicate = False
                for ff in frontal_faces:
                    iou = _compute_iou(pf['bbox'], ff['bbox'])
                    if iou > 0.5:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    all_faces.append(pf)

        # 3. Apply NMS to remove overlapping detections
        if len(all_faces) > 1:
            boxes = [f['bbox'] for f in all_faces]
            scores = [f['score'] for f in all_faces]
            keep_indices = _nms(boxes, scores, threshold=0.3)
            all_faces = [all_faces[i] for i in keep_indices]

        return all_faces

    def release(self):
        pass
