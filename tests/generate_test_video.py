"""
Generate synthetic test videos with faces and various mosaic/blur effects.
Creates videos that simulate:
1. Properly blurred faces
2. Insufficiently blurred faces (partial protection)
3. Faces with mosaic that has gaps
4. Faces that are completely unblurred in some frames
5. Multiple people with different protection levels
"""

import cv2
import numpy as np
import os


def draw_face(image, center, size, seed=42):
    """Draw a cartoon-like face on the image."""
    rng = np.random.RandomState(seed)
    cx, cy = center
    r = size // 2

    # Face shape (ellipse)
    skin_colors = [(255, 220, 180), (255, 210, 170), (240, 190, 150), (255, 230, 200)]
    skin = skin_colors[rng.randint(len(skin_colors))]
    cv2.ellipse(image, (cx, cy), (r, int(r * 1.2)), 0, 0, 360, skin, -1)

    # Eyes
    eye_y = cy - r // 4
    eye_offset = r // 3
    eye_r = max(3, r // 8)
    eye_white = (255, 255, 255)
    pupil_colors = [(50, 50, 150), (100, 50, 50), (50, 100, 50), (80, 60, 40)]
    pupil_color = pupil_colors[rng.randint(len(pupil_colors))]

    for dx in [-eye_offset, eye_offset]:
        cv2.circle(image, (cx + dx, eye_y), eye_r, eye_white, -1)
        cv2.circle(image, (cx + dx + rng.randint(-1, 2), eye_y), eye_r // 2, pupil_color, -1)

    # Nose
    nose_y = cy + r // 10
    cv2.ellipse(image, (cx, nose_y), (r // 8, r // 6), 0, 0, 360, (200, 150, 120), -1)

    # Mouth
    mouth_y = cy + r // 3
    lip_color = (180, 100, 100)
    cv2.ellipse(image, (cx, mouth_y), (r // 3, r // 6), 0, 0, 180, lip_color, 2)

    # Hair
    hair_colors = [(30, 30, 30), (60, 40, 20), (80, 60, 40), (120, 80, 60), (200, 150, 100)]
    hair_color = hair_colors[rng.randint(len(hair_colors))]
    hair_y = cy - int(r * 1.0)
    cv2.ellipse(image, (cx, hair_y), (r + 5, r // 2), 0, 0, 180, hair_color, -1)
    for i in range(5):
        hx = cx + rng.randint(-r, r + 1)
        hy = hair_y + rng.randint(-10, 10)
        cv2.circle(image, (hx, hy), rng.randint(3, 8), hair_color, -1)

    # Eyebrows
    brow_y = eye_y - eye_r - 2
    for dx in [-eye_offset, eye_offset]:
        cv2.line(image, (cx + dx - 6, brow_y), (cx + dx + 6, brow_y - 2), hair_color, 2)

    return image


def apply_blur(image, bbox, strength=51):
    """Apply gaussian blur to a region."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return image

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return image

    ksize = strength if strength % 2 == 1 else strength + 1
    blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
    image[y1:y2, x1:x2] = blurred
    return image


def apply_mosaic(image, bbox, block_size=12):
    """Apply pixelation/mosaic to a region."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return image

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return image

    rh, rw = roi.shape[:2]
    small = cv2.resize(roi, (rw // block_size, rh // block_size), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = mosaic
    return image


def apply_black_block(image, bbox):
    """Apply solid black block."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return image


def generate_test_video_1(output_path, duration_sec=8, fps=30):
    """
    Test video 1: Single face with blur, then blur fails for a few frames.
    Simulates: blur effect that disappears intermittently.
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    face_center = (width // 2, height // 2 + 20)
    face_size = 120
    bbox = [face_center[0] - face_size, face_center[1] - face_size,
            face_center[0] + face_size, face_center[1] + face_size]

    for frame_idx in range(total_frames):
        # Background
        frame = np.full((height, width, 3), (240, 240, 245), dtype=np.uint8)

        # Draw room-like background
        cv2.rectangle(frame, (0, 0), (width, height // 2), (200, 210, 220), -1)
        cv2.rectangle(frame, (0, height // 2), (width, height), (180, 170, 160), -1)

        # Draw face
        draw_face(frame, face_center, face_size, seed=42)

        # Apply effects with intentional gaps
        time_sec = frame_idx / fps

        if time_sec < 2.0:
            # Frames 0-2s: properly blurred
            frame = apply_blur(frame, bbox, strength=51)
        elif 2.0 <= time_sec < 2.5:
            # Frames 2.0-2.5s: NO blur (face exposed!) - simulates tracking failure
            pass
        elif 2.5 <= time_sec < 5.0:
            # Frames 2.5-5.0s: properly blurred again
            frame = apply_blur(frame, bbox, strength=51)
        elif 5.0 <= time_sec < 5.8:
            # Frames 5.0-5.8s: weak blur (insufficient) - simulates weak blur strength
            frame = apply_blur(frame, bbox, strength=15)
        elif 5.8 <= time_sec < 6.3:
            # Frames 5.8-6.3s: NO blur (face exposed again!)
            pass
        else:
            # Frames 6.3-8.0s: properly blurred
            frame = apply_blur(frame, bbox, strength=51)

        # Timestamp
        ts = f"{int(time_sec // 60):02d}:{int(time_sec % 60):02d}.{int((time_sec % 1) * 1000):03d}"
        cv2.putText(frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)

        out.write(frame)

    out.release()
    print(f"Generated: {output_path} ({total_frames} frames, {duration_sec}s)")


def generate_test_video_2(output_path, duration_sec=10, fps=30):
    """
    Test video 2: Two people, one properly blurred, one completely unblurred.
    Simulates: forgot to blur one person.
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    face1_center = (width // 3, height // 2)
    face2_center = (2 * width // 3, height // 2)
    face_size = 100

    bbox1 = [face1_center[0] - face_size, face1_center[1] - face_size,
             face1_center[0] + face_size, face1_center[1] + face_size]
    bbox2 = [face2_center[0] - face_size, face2_center[1] - face_size,
             face2_center[0] + face_size, face2_center[1] + face_size]

    for frame_idx in range(total_frames):
        frame = np.full((height, width, 3), (230, 230, 235), dtype=np.uint8)

        # Background
        cv2.rectangle(frame, (0, 0), (width, height // 2), (210, 220, 230), -1)

        # Draw both faces
        draw_face(frame, face1_center, face_size, seed=42)
        draw_face(frame, face2_center, face_size, seed=99)

        # Person 1: always blurred (properly protected)
        frame = apply_blur(frame, bbox1, strength=51)

        # Person 2: NEVER blurred (completely unprotected!)
        # This simulates forgetting to blur one person
        pass

        # Add labels
        cv2.putText(frame, "Person 1 (blurred)", (face1_center[0] - 80, face1_center[1] + face_size + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
        cv2.putText(frame, "Person 2 (UNBLURRED!)", (face2_center[0] - 90, face2_center[1] + face_size + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

        time_sec = frame_idx / fps
        ts = f"{int(time_sec // 60):02d}:{int(time_sec % 60):02d}.{int((time_sec % 1) * 1000):03d}"
        cv2.putText(frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)

        out.write(frame)

    out.release()
    print(f"Generated: {output_path} ({total_frames} frames, {duration_sec}s)")


def generate_test_video_3(output_path, duration_sec=8, fps=30):
    """
    Test video 3: Face with mosaic that has varying block sizes.
    Some frames have large blocks (insufficient), some small (good).
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    face_center = (width // 2, height // 2)
    face_size = 120
    bbox = [face_center[0] - face_size, face_center[1] - face_size,
            face_center[0] + face_size, face_center[1] + face_size]

    for frame_idx in range(total_frames):
        frame = np.full((height, width, 3), (245, 240, 235), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (width, height // 2), (225, 220, 215), -1)

        draw_face(frame, face_center, face_size, seed=42)

        time_sec = frame_idx / fps

        if time_sec < 2.5:
            # Good mosaic (small blocks)
            frame = apply_mosaic(frame, bbox, block_size=8)
        elif 2.5 <= time_sec < 4.5:
            # Bad mosaic (large blocks - can still see face!)
            frame = apply_mosaic(frame, bbox, block_size=24)
        elif 4.5 <= time_sec < 5.5:
            # No mosaic at all!
            pass
        elif 5.5 <= time_sec < 6.5:
            # Black block
            frame = apply_black_block(frame, bbox)
        else:
            # Good mosaic again
            frame = apply_mosaic(frame, bbox, block_size=8)

        # Label current effect
        effects = {
            (0, 2.5): "Mosaic (good)",
            (2.5, 4.5): "Mosaic (WEAK - large blocks!)",
            (4.5, 5.5): "NO EFFECT - face exposed!",
            (5.5, 6.5): "Black block",
            (6.5, 8.0): "Mosaic (good)",
        }
        for (s, e), label in effects.items():
            if s <= time_sec < e:
                color = (0, 150, 0) if "good" in label else ((0, 0, 200) if "exposed" in label or "WEAK" in label else (200, 150, 0))
                cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                break

        ts = f"{int(time_sec // 60):02d}:{int(time_sec % 60):02d}.{int((time_sec % 1) * 1000):03d}"
        cv2.putText(frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)

        out.write(frame)

    out.release()
    print(f"Generated: {output_path} ({total_frames} frames, {duration_sec}s)")


def generate_test_video_4(output_path, duration_sec=8, fps=30):
    """
    Test video 4: Face with black block that shifts/disappears intermittently.
    Simulates tracking-based blur that loses track.
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    face_center = (width // 2, height // 2)
    face_size = 110
    bbox = [face_center[0] - face_size, face_center[1] - face_size,
            face_center[0] + face_size, face_center[1] + face_size]

    for frame_idx in range(total_frames):
        frame = np.full((height, width, 3), (235, 235, 240), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (width, height // 2), (215, 215, 220), -1)

        # Slight movement
        offset_x = int(10 * np.sin(frame_idx * 0.1))
        offset_y = int(5 * np.cos(frame_idx * 0.15))
        moved_center = (face_center[0] + offset_x, face_center[1] + offset_y)
        moved_bbox = [moved_center[0] - face_size, moved_center[1] - face_size,
                      moved_center[0] + face_size, moved_center[1] + face_size]

        draw_face(frame, moved_center, face_size, seed=42)

        time_sec = frame_idx / fps

        if time_sec < 1.5:
            frame = apply_black_block(frame, moved_bbox)
        elif 1.5 <= time_sec < 2.2:
            # Block shifts (wrong position) - face partially exposed
            wrong_bbox = [b + 30 for b in moved_bbox]
            frame = apply_black_block(frame, wrong_bbox)
        elif 2.2 <= time_sec < 3.5:
            frame = apply_black_block(frame, moved_bbox)
        elif 3.5 <= time_sec < 4.0:
            # Block disappears completely!
            pass
        elif 4.0 <= time_sec < 6.0:
            frame = apply_black_block(frame, moved_bbox)
        elif 6.0 <= time_sec < 6.5:
            # Block too small
            small_bbox = [moved_center[0] - face_size//2, moved_center[1] - face_size//2,
                          moved_center[0] + face_size//2, moved_center[1] + face_size//2]
            frame = apply_black_block(frame, small_bbox)
        else:
            frame = apply_black_block(frame, moved_bbox)

        ts = f"{int(time_sec // 60):02d}:{int(time_sec % 60):02d}.{int((time_sec % 1) * 1000):03d}"
        cv2.putText(frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)

        out.write(frame)

    out.release()
    print(f"Generated: {output_path} ({total_frames} frames, {duration_sec}s)")


def generate_all_test_videos(output_dir):
    """Generate all test videos."""
    os.makedirs(output_dir, exist_ok=True)

    generate_test_video_1(os.path.join(output_dir, "test_blur_gaps.mp4"), duration_sec=8)
    generate_test_video_2(os.path.join(output_dir, "test_two_people_one_unblurred.mp4"), duration_sec=10)
    generate_test_video_3(os.path.join(output_dir, "test_mosaic_weak.mp4"), duration_sec=8)
    generate_test_video_4(os.path.join(output_dir, "test_black_block_shifts.mp4"), duration_sec=8)

    print(f"\nAll test videos generated in: {output_dir}")
    print("Files:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.mp4'):
            size_mb = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "test_videos")
    generate_all_test_videos(out_dir)
