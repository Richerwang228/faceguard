import numpy as np
from typing import List, Dict, Optional


def compute_iou(box1: List[float], box2: List[float]) -> float:
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


def _predict_bbox(bbox: List[float], velocity: List[float]) -> List[float]:
    """Predict next bbox position using velocity."""
    return [
        bbox[0] + velocity[0],
        bbox[1] + velocity[1],
        bbox[2] + velocity[0],
        bbox[3] + velocity[1],
    ]


class FaceTracker:
    """IOU-based face tracker with motion prediction for fast-moving objects."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 5,
        prediction_iou_threshold: float = 0.15,
        velocity_smooth: float = 0.7,
    ):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.prediction_iou_threshold = prediction_iou_threshold
        self.velocity_smooth = velocity_smooth
        self.tracks = []
        self.next_id = 1

    def _compute_velocity(self, track: dict) -> List[float]:
        """Compute smoothed velocity from track history."""
        if 'velocity' not in track:
            return [0.0, 0.0]
        return track['velocity']

    def _update_velocity(self, track: dict, new_bbox: List[float]) -> None:
        """Update track velocity with exponential smoothing."""
        old_bbox = track['bbox']
        dx = new_bbox[0] - old_bbox[0]
        dy = new_bbox[1] - old_bbox[1]

        if 'velocity' not in track:
            track['velocity'] = [float(dx), float(dy)]
        else:
            track['velocity'][0] = (
                self.velocity_smooth * track['velocity'][0]
                + (1 - self.velocity_smooth) * dx
            )
            track['velocity'][1] = (
                self.velocity_smooth * track['velocity'][1]
                + (1 - self.velocity_smooth) * dy
            )

    def update(self, detections: List[dict], frame_idx: int) -> List[dict]:
        """
        Update tracks with new detections.
        Uses motion prediction when direct IOU matching fails.
        Returns list of detection dicts with 'track_id' added.
        """
        # Increment missed count for all tracks
        for track in self.tracks:
            track['missed'] += 1

        # Build predicted bboxes for all active tracks
        for track in self.tracks:
            if track['missed'] <= self.max_missed and 'velocity' in track:
                track['predicted_bbox'] = _predict_bbox(
                    track['bbox'], track['velocity']
                )
            else:
                track['predicted_bbox'] = track['bbox']

        # Match detections to existing tracks (two-pass)
        matched_detections = set()
        matched_tracks = set()

        # Pass 1: Direct IOU matching
        for i, det in enumerate(detections):
            if i in matched_detections:
                continue
            best_iou = 0.0
            best_track_idx = -1
            for ti, track in enumerate(self.tracks):
                if ti in matched_tracks or track['missed'] > self.max_missed:
                    continue
                iou = compute_iou(det['bbox'], track['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_idx = ti

            if best_track_idx >= 0:
                track = self.tracks[best_track_idx]
                self._update_velocity(track, det['bbox'])
                track['bbox'] = det['bbox']
                track['score'] = det['score']
                track['missed'] = 0
                track['frames_seen'] += 1
                track['last_frame'] = frame_idx
                det['track_id'] = track['id']
                matched_detections.add(i)
                matched_tracks.add(best_track_idx)

        # Pass 2: Predicted bbox matching for fast-moving objects
        for i, det in enumerate(detections):
            if i in matched_detections:
                continue
            best_iou = 0.0
            best_track_idx = -1
            for ti, track in enumerate(self.tracks):
                if ti in matched_tracks or track['missed'] > self.max_missed:
                    continue
                iou = compute_iou(det['bbox'], track['predicted_bbox'])
                if iou > best_iou and iou > self.prediction_iou_threshold:
                    best_iou = iou
                    best_track_idx = ti

            if best_track_idx >= 0:
                track = self.tracks[best_track_idx]
                self._update_velocity(track, det['bbox'])
                track['bbox'] = det['bbox']
                track['score'] = det['score']
                track['missed'] = 0
                track['frames_seen'] += 1
                track['last_frame'] = frame_idx
                det['track_id'] = track['id']
                matched_detections.add(i)
                matched_tracks.add(best_track_idx)

        # Pass 3: Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i in matched_detections:
                continue
            new_track = {
                'id': self.next_id,
                'bbox': det['bbox'],
                'score': det['score'],
                'missed': 0,
                'frames_seen': 1,
                'first_frame': frame_idx,
                'last_frame': frame_idx,
                'frames': [],
                'velocity': [0.0, 0.0],
                'predicted_bbox': det['bbox'],
            }
            self.tracks.append(new_track)
            det['track_id'] = self.next_id
            self.next_id += 1

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t['missed'] <= self.max_missed]

        return detections

    def get_active_tracks(self) -> List[Dict]:
        """Get currently active (visible) tracks."""
        return [t for t in self.tracks if t['missed'] == 0]

    def get_all_tracks(self) -> List[Dict]:
        """Get all tracks including recently lost ones."""
        return self.tracks

    def reset(self):
        self.tracks = []
        self.next_id = 1
