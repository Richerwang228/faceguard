import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from .detector import FaceDetector
from .tracker import FaceTracker
from .mosaic_checker import MosaicChecker


class VideoProcessor:
    """Main video processing pipeline: detect faces, track across frames, check mosaic quality."""

    # Supported video codecs with fallback chain
    _SUPPORTED_FOURCC_FALLBACKS = [
        ('mp4v', '.mp4'),
        ('avc1', '.mp4'),
        ('H264', '.mp4'),
        ('hevc', '.mp4'),
        ('hvc1', '.mp4'),
        ('VP90', '.webm'),
        ('VP80', '.webm'),
        ('XVID', '.avi'),
    ]

    def __init__(
        self,
        sample_rate: int = 5,
        min_face_size: int = 64,
        blur_threshold: float = 80.0,
        iou_threshold: float = 0.3,
        max_missed: int = 5,
        min_detection_confidence: float = 0.5,
        adaptive_sampling: bool = True,
    ):
        self.sample_rate = sample_rate
        self.min_face_size = min_face_size
        self.adaptive_sampling = adaptive_sampling
        self.detector = FaceDetector(min_detection_confidence)
        self.tracker = FaceTracker(iou_threshold, max_missed)
        self.checker = MosaicChecker(blur_threshold)
        self.frame_results = []
        self.all_tracks_history = []

    def process(
        self,
        video_path: str,
        progress_callback=None,
    ) -> Dict:
        """
        Process a video file.
        Returns dict with full analysis results.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        # Smart adaptive sampling based on video duration
        effective_sample_rate = self.sample_rate
        if self.adaptive_sampling and duration > 0:
            if duration <= 5.0:
                # Short video (<5s): high sampling for accuracy
                effective_sample_rate = max(self.sample_rate, 10)
            elif duration <= 30.0:
                # Medium video (5-30s): use configured rate
                effective_sample_rate = self.sample_rate
            elif duration <= 120.0:
                # Long video (30s-2min): slightly reduce
                effective_sample_rate = max(3, self.sample_rate - 1)
            else:
                # Very long video (>2min): reduce to save time
                effective_sample_rate = max(2, self.sample_rate - 2)

        # Calculate which frames to sample
        sample_interval = int(fps / effective_sample_rate) if fps > effective_sample_rate else 1
        target_frames = list(range(0, total_frames, sample_interval))
        num_target = len(target_frames)

        self.frame_results = []
        self.tracker.reset()
        track_history = {}  # track_id -> list of per-frame results

        frame_idx = 0
        target_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if target_idx < num_target and frame_idx == target_frames[target_idx]:
                    timestamp = frame_idx / fps
                    timestamp_str = self._format_time(timestamp)

                    # Detect faces
                    detections = self.detector.detect(frame)

                    # Filter by minimum face size
                    detections = [
                        d for d in detections
                        if (d['bbox'][2] - d['bbox'][0]) >= self.min_face_size
                        and (d['bbox'][3] - d['bbox'][1]) >= self.min_face_size
                    ]

                    # Track across frames
                    tracked = self.tracker.update(detections, frame_idx)

                    # Check mosaic quality for each tracked face
                    frame_problems = []
                    for det in tracked:
                        track_id = det['track_id']
                        check_result = self.checker.check_region(frame, det['bbox'])

                        record = {
                            'frame_idx': frame_idx,
                            'timestamp': timestamp,
                            'timestamp_str': timestamp_str,
                            'track_id': track_id,
                            'bbox': det['bbox'],
                            'score': det['score'],
                            'is_obscured': check_result['is_obscured'],
                            'methods': check_result['methods'],
                            'confidence': check_result['confidence'],
                            'details': check_result['details'],
                        }

                        if track_id not in track_history:
                            track_history[track_id] = []
                        track_history[track_id].append(record)

                        # If not obscured, it's a problem frame
                        if not check_result['is_obscured']:
                            frame_problems.append(record)

                    self.frame_results.append({
                        'frame_idx': frame_idx,
                        'timestamp': timestamp,
                        'timestamp_str': timestamp_str,
                        'num_faces': len(tracked),
                        'num_problems': len(frame_problems),
                        'detections': tracked,
                        'problems': frame_problems,
                    })

                    target_idx += 1

                    # Progress callback
                    if progress_callback:
                        progress = target_idx / num_target
                        progress_callback(progress, target_idx, num_target, timestamp_str)

                frame_idx += 1
        finally:
            cap.release()
            self.detector.release()

        # Analyze track statistics
        person_stats = self._analyze_tracks(track_history, duration)

        # Collect problem frames
        problem_frames = []
        for fr in self.frame_results:
            for p in fr['problems']:
                problem_frames.append({
                    'frame_idx': p['frame_idx'],
                    'timestamp': p['timestamp'],
                    'timestamp_str': p['timestamp_str'],
                    'track_id': p['track_id'],
                    'bbox': p['bbox'],
                    'score': p['score'],
                })

        # Determine overall status
        total_sampled = len(self.frame_results)
        problem_count = len(problem_frames)
        problem_ratio = problem_count / total_sampled if total_sampled > 0 else 0

        if problem_count == 0:
            status = 'passed'
            risk_level = 'none'
        elif problem_ratio < 0.005:
            status = 'warning'
            risk_level = 'low'
        elif problem_ratio < 0.02:
            status = 'failed'
            risk_level = 'medium'
        else:
            status = 'failed'
            risk_level = 'high'

        return {
            'video_info': {
                'path': video_path,
                'filename': os.path.basename(video_path),
                'duration': duration,
                'duration_str': self._format_time(duration),
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'sampled_frames': total_sampled,
                'sample_rate': effective_sample_rate,
                'adaptive_sampling': self.adaptive_sampling,
            },
            'summary': {
                'status': status,
                'risk_level': risk_level,
                'total_persons': len(person_stats),
                'problem_frames_count': problem_count,
                'problem_frames_ratio': problem_ratio,
                'sampled_frames': total_sampled,
            },
            'persons': person_stats,
            'problem_frames': problem_frames,
            'frame_results': self.frame_results,
        }

    def _analyze_tracks(self, track_history: Dict, duration: float) -> List[Dict]:
        """Analyze each track to determine if it's protected, at risk, or unprotected."""
        persons = []

        for track_id, records in track_history.items():
            total = len(records)
            protected = sum(1 for r in records if r['is_obscured'])
            unprotected = total - protected
            ratio = unprotected / total if total > 0 else 0

            # Determine status based on protection ratio
            if protected == 0:
                status = 'unprotected'
                status_label = '未打码'
            elif unprotected == 0:
                status = 'protected'
                status_label = '已保护'
            else:
                status = 'partial_risk'
                status_label = '漏打风险'

            # Find problem timestamps
            problem_records = [r for r in records if not r['is_obscured']]
            problem_timestamps = [
                {
                    'timestamp': r['timestamp'],
                    'timestamp_str': r['timestamp_str'],
                    'frame_idx': r['frame_idx'],
                    'bbox': r['bbox'],
                    'confidence': r['score'],
                }
                for r in problem_records
            ]

            first_ts = records[0]['timestamp_str'] if records else '00:00:00'
            last_ts = records[-1]['timestamp_str'] if records else '00:00:00'

            # Detected mosaic methods
            all_methods = set()
            for r in records:
                all_methods.update(r['methods'])

            persons.append({
                'track_id': track_id,
                'status': status,
                'status_label': status_label,
                'frames_total': total,
                'frames_protected': protected,
                'frames_unprotected': unprotected,
                'unprotected_ratio': ratio,
                'first_appearance': first_ts,
                'last_appearance': last_ts,
                'problem_timestamps': problem_timestamps,
                'detected_methods': sorted(list(all_methods)),
            })

        # Sort: unprotected first, then partial, then protected
        order = {'unprotected': 0, 'partial_risk': 1, 'protected': 2}
        persons.sort(key=lambda p: order.get(p['status'], 3))
        return persons

    def extract_screenshots(
        self,
        video_path: str,
        problem_frames: List[Dict],
        output_dir: str,
        max_screenshots: int = 20,
    ) -> List[str]:
        """Extract annotated screenshots for problem frames."""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        saved = []

        # Deduplicate by frame_idx
        seen_frames = set()
        unique_problems = []
        for p in problem_frames:
            if p['frame_idx'] not in seen_frames and len(unique_problems) < max_screenshots:
                seen_frames.add(p['frame_idx'])
                unique_problems.append(p)

        for p in unique_problems:
            cap.set(cv2.CAP_PROP_POS_FRAMES, p['frame_idx'])
            ret, frame = cap.read()
            if not ret:
                continue

            # Draw annotation
            x1, y1, x2, y2 = p['bbox']
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Red dashed rectangle for detected face
            self._draw_dashed_rect(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Label
            label = f"Person #{p['track_id']} - {p['timestamp_str']}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Warning text
            warning = "FACE EXPOSED!"
            tw = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
            cv2.rectangle(frame, (10, 10), (10 + tw + 20, 45), (0, 0, 255), -1)
            cv2.putText(frame, warning, (20, 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            filename = f"frame_{p['frame_idx']:06d}_person_{p['track_id']}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved.append(filepath)

        cap.release()
        return saved

    def _draw_dashed_rect(self, img, pt1, pt2, color, thickness, dash_len=10):
        """Draw a dashed rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Top edge
        for x in range(x1, x2, dash_len * 2):
            cv2.line(img, (x, y1), (min(x + dash_len, x2), y1), color, thickness)
        # Bottom edge
        for x in range(x1, x2, dash_len * 2):
            cv2.line(img, (x, y2), (min(x + dash_len, x2), y2), color, thickness)
        # Left edge
        for y in range(y1, y2, dash_len * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_len, y2)), color, thickness)
        # Right edge
        for y in range(y1, y2, dash_len * 2):
            cv2.line(img, (x2, y), (x2, min(y + dash_len, y2)), color, thickness)

    @classmethod
    def get_supported_codecs(cls) -> List[Dict[str, str]]:
        """Return list of supported video codecs and their file extensions."""
        return [
            {'codec': c, 'extension': e} for c, e in cls._SUPPORTED_FOURCC_FALLBACKS
        ]

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS.mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
        return f"{minutes:02d}:{secs:02d}.{ms:03d}"
