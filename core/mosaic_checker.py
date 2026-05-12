import cv2
import numpy as np
from typing import List, Dict


class MosaicChecker:
    """
    Detects whether a face region is properly obscured/blurred/mosaicked.
    Uses multiple texture analysis metrics for robust detection.
    """

    def __init__(self, clarity_threshold: float = 50.0):
        """
        Args:
            clarity_threshold: Score above this = clearly visible face (not obscured)
        """
        self.clarity_threshold = clarity_threshold

    def check_region(self, image: np.ndarray, bbox: List[int]) -> Dict:
        """
        Check a face region for mosaic/blur coverage.
        Returns dict with detection results.
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return {'is_obscured': False, 'methods': [], 'confidence': 0.0, 'clarity_score': 0.0}

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return {'is_obscured': False, 'methods': [], 'confidence': 0.0, 'clarity_score': 0.0}

        # Compute primary clarity metrics
        clarity = self._compute_clarity(roi)
        clarity_score = clarity['score']

        # Run all specialized obscuration detectors
        methods = []
        confidences = []

        # 1. Check for blur (low clarity)
        if clarity_score < self.clarity_threshold:
            methods.append('blur')
            confidences.append(1.0 - clarity_score / self.clarity_threshold)

        # 2. Check for pixelation/mosaic
        mosaic_score = self._check_mosaic(roi)
        if mosaic_score['is_mosaic']:
            methods.append('mosaic')
            confidences.append(mosaic_score['confidence'])

        # 3. Check for solid color block
        solid_score = self._check_solid_block(roi)
        if solid_score['is_solid']:
            methods.append('solid')
            confidences.append(solid_score['confidence'])

        # 4. Check for sticker
        sticker_score = self._check_sticker(roi)
        if sticker_score['is_sticker']:
            methods.append('sticker')
            confidences.append(sticker_score['confidence'])

        # DECISION: If any specialized detector finds obscuration, face is protected
        # This takes priority over clarity (e.g. mosaic may have high edge score)
        is_obscured = len(methods) > 0

        # EXCEPTION: Very high clarity + no strong special detector = likely unobscured
        # Only override if clarity is very high and the only method is weak blur
        if clarity_score >= self.clarity_threshold + 30 and len(methods) == 1 and methods[0] == 'blur':
            is_obscured = False
            methods = []
            confidences = []

        avg_confidence = max(confidences) if confidences else 0.0

        return {
            'is_obscured': is_obscured,
            'methods': methods,
            'confidence': avg_confidence,
            'clarity_score': clarity_score,
            'details': {
                'clarity': clarity,
                'mosaic': mosaic_score,
                'solid': solid_score,
                'sticker': sticker_score
            }
        }

    def _compute_clarity(self, roi: np.ndarray) -> Dict:
        """
        Compute overall clarity score of a region.
        Higher = more clear/detail (unobscured face).
        Lower = more obscured/blurred.
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < 8 or w < 8:
            return {'score': 0.0, 'laplacian': 0.0, 'sobel': 0.0, 'contrast': 0.0, 'block_var': 0.0}

        # 1. Laplacian variance (edge sharpness)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(lap.var())

        # 2. Sobel gradient magnitude
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))

        # 3. Local contrast
        local_std = float(np.std(gray))

        # 4. Block variance (variance of variances in 8x8 blocks)
        block_vars = []
        block_size = 8
        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                block = gray[by:by+block_size, bx:bx+block_size]
                block_vars.append(np.var(block))
        block_var_mean = float(np.mean(block_vars)) if block_vars else 0.0

        # Normalize each metric
        # Empirical calibration from testing real faces:
        # Clear face: lap_var ~200-1000, sobel ~20-80, local_std ~30-80, block_var_mean ~300-800
        # Blurred face: lap_var ~10-50, sobel ~2-10, local_std ~5-20, block_var_mean ~20-100
        lap_norm = min(100, lap_var / 5)
        sobel_norm = min(100, sobel_mag * 2)
        contrast_norm = min(100, local_std * 2)
        block_var_norm = min(100, block_var_mean / 5)

        # Combined score
        score = lap_norm * 0.3 + sobel_norm * 0.25 + contrast_norm * 0.2 + block_var_norm * 0.25

        return {
            'score': score,
            'laplacian': lap_var,
            'sobel': sobel_mag,
            'contrast': local_std,
            'block_var': block_var_mean,
        }

    def _check_mosaic(self, roi: np.ndarray) -> Dict:
        """
        Check for pixelation/mosaic pattern.
        True mosaic has many low-variance blocks (same value within block)
        but not completely uniform (has some structure).
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        if h < 16 or w < 16:
            return {'is_mosaic': False, 'low_var_ratio': 0.0, 'mean_var': 0.0, 'confidence': 0.0}

        block_vars = []
        block_size = 8
        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                block = gray[by:by+block_size, bx:bx+block_size]
                block_vars.append(np.var(block))

        block_vars = np.array(block_vars)
        low_var_ratio = float(np.mean(block_vars < 5))
        mean_var = float(np.mean(block_vars))

        # Mosaic: many low-variance blocks but some structure remains
        # Pure solid color would have low_var_ratio near 1.0
        # Clear face would have low_var_ratio near 0
        # Mosaic is in between: many blocks with low variance, but not all
        is_mosaic = 0.2 < low_var_ratio < 0.8 and mean_var < 400
        confidence = min(1.0, low_var_ratio * 2) if is_mosaic else 0.0

        return {
            'is_mosaic': is_mosaic,
            'low_var_ratio': low_var_ratio,
            'mean_var': mean_var,
            'confidence': confidence
        }

    def _check_solid_block(self, roi: np.ndarray) -> Dict:
        """Check if region is a solid color block."""
        h, w = roi.shape[:2]
        if h < 4 or w < 4:
            return {'is_solid': False, 'uniformity': 0.0, 'confidence': 0.0}

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_std = float(np.std(gray))

        # Also check color channels
        b_std = float(np.std(roi[:, :, 0]))
        g_std = float(np.std(roi[:, :, 1]))
        r_std = float(np.std(roi[:, :, 2]))
        color_std = (b_std + g_std + r_std) / 3

        is_solid = gray_std < 8.0 and color_std < 10.0
        confidence = min(1.0, (15.0 - gray_std) / 15.0) if is_solid else 0.0

        return {
            'is_solid': is_solid,
            'gray_std': gray_std,
            'color_std': color_std,
            'confidence': confidence
        }

    def _check_sticker(self, roi: np.ndarray) -> Dict:
        """Check if region has sticker-like characteristics."""
        h, w = roi.shape[:2]
        if h < 16 or w < 16:
            return {'is_sticker': False, 'edge_score': 0.0, 'confidence': 0.0}

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.sum(edges > 0) / edges.size)

        dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        interior_mask = dilated_edges == 0
        if np.sum(interior_mask) > 100:
            interior_std = float(np.std(gray[interior_mask]))
        else:
            interior_std = 999.0

        is_sticker = 0.02 < edge_ratio < 0.3 and interior_std < 30.0
        confidence = min(1.0, edge_ratio * 10) if is_sticker else 0.0

        return {
            'is_sticker': is_sticker,
            'edge_ratio': edge_ratio,
            'interior_std': interior_std,
            'confidence': confidence
        }
