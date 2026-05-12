"""
Generate test videos using REAL face images (from thispersondoesnotexist.com).
This ensures Haar Cascade / face detectors can actually detect faces.
"""

import cv2
import numpy as np
import os


def load_face_image(path, target_size=(200, 250)):
    """Load and resize a face image with proper aspect ratio."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")

    # Resize to target size
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Create circular mask for softer edges
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (w // 2 - 5, h // 2 - 5)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # Apply mask
    masked = img.copy()
    masked = cv2.bitwise_and(masked, masked, mask=mask)

    return masked, mask


def composite_face_onto_background(face_img, face_mask, bg, center, scale=1.0):
    """Composite a face image onto a background at given center."""
    h, w = face_img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    face_resized = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(face_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cx, cy = center
    x1 = cx - new_w // 2
    y1 = cy - new_h // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    bg_h, bg_w = bg.shape[:2]

    # Clamp to background bounds
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(bg_w, x2), min(bg_h, y2)

    if fx2 <= fx1 or fy2 <= fy1:
        return bg

    # Face region offsets
    foff_x = fx1 - x1
    foff_y = fy1 - y1
    foff_x2 = foff_x + (fx2 - fx1)
    foff_y2 = foff_y + (fy2 - fy1)

    face_crop = face_resized[foff_y:foff_y2, foff_x:foff_x2]
    mask_crop = mask_resized[foff_y:foff_y2, foff_x:foff_x2]

    # Composite
    roi = bg[fy1:fy2, fx1:fx2]
    mask_3ch = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR) / 255.0
    composite = (face_crop * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
    bg[fy1:fy2, fx1:fx2] = composite

    return bg


def get_face_bbox(center, face_size, scale=1.0):
    """Get bounding box for face."""
    w, h = int(face_size[0] * scale), int(face_size[1] * scale)
    cx, cy = center
    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]


def apply_blur(image, bbox, strength=51):
    """Apply gaussian blur to a region."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
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
    """Apply pixelation/mosaic."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return image
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return image
    rh, rw = roi.shape[:2]
    small = cv2.resize(roi, (max(1, rw // block_size), max(1, rh // block_size)), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = mosaic
    return image


def apply_black_block(image, bbox):
    """Apply solid black block."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return image


def create_background(width, height, frame_idx=0):
    """Create a simple room-like background."""
    bg = np.full((height, width, 3), (240, 235, 230), dtype=np.uint8)

    # Wall
    cv2.rectangle(bg, (0, 0), (width, height // 2 + 30), (220, 215, 210), -1)

    # Floor
    cv2.rectangle(bg, (0, height // 2 + 30), (width, height), (190, 185, 180), -1)

    # Add some noise for texture
    noise = np.random.randint(-5, 5, (height, width, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return bg


def add_timestamp(frame, time_sec):
    """Add timestamp to frame."""
    minutes = int(time_sec // 60)
    seconds = int(time_sec % 60)
    ms = int((time_sec % 1) * 1000)
    ts = f"{minutes:02d}:{seconds:02d}.{ms:03d}"
    cv2.putText(frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
    return frame


def generate_test_video_1(output_path, face_image_path, duration_sec=8, fps=30):
    """Test 1: Single face with blur, blur fails intermittently."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    face_img, face_mask = load_face_image(face_image_path, (200, 250))
    face_size = (200, 250)
    center = (width // 2, height // 2 + 20)

    total_frames = duration_sec * fps

    for frame_idx in range(total_frames):
        bg = create_background(width, height, frame_idx)
        frame = composite_face_onto_background(face_img, face_mask, bg, center, scale=1.0)
        bbox = get_face_bbox(center, face_size, scale=1.0)

        time_sec = frame_idx / fps

        if time_sec < 2.0:
            frame = apply_blur(frame, bbox, strength=51)
        elif 2.0 <= time_sec < 2.5:
            pass  # NO blur - face exposed!
        elif 2.5 <= time_sec < 5.0:
            frame = apply_blur(frame, bbox, strength=51)
        elif 5.0 <= time_sec < 5.8:
            frame = apply_blur(frame, bbox, strength=15)  # Weak blur
        elif 5.8 <= time_sec < 6.3:
            pass  # NO blur again!
        else:
            frame = apply_blur(frame, bbox, strength=51)

        frame = add_timestamp(frame, time_sec)
        out.write(frame)

    out.release()
    print(f"Generated: {output_path}")


def generate_test_video_2(output_path, face1_path, face2_path, duration_sec=10, fps=30):
    """Test 2: Two people, one blurred, one completely unblurred."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    face1_img, face1_mask = load_face_image(face1_path, (180, 220))
    face2_img, face2_mask = load_face_image(face2_path, (180, 220))
    face_size = (180, 220)

    center1 = (width // 3, height // 2)
    center2 = (2 * width // 3, height // 2)
    bbox1 = get_face_bbox(center1, face_size, scale=1.0)
    bbox2 = get_face_bbox(center2, face_size, scale=1.0)

    total_frames = duration_sec * fps

    for frame_idx in range(total_frames):
        bg = create_background(width, height, frame_idx)
        frame = composite_face_onto_background(face1_img, face1_mask, bg, center1, scale=1.0)
        frame = composite_face_onto_background(face2_img, face2_mask, frame, center2, scale=1.0)

        # Person 1: always blurred
        frame = apply_blur(frame, bbox1, strength=51)
        # Person 2: NEVER blurred

        time_sec = frame_idx / fps
        frame = add_timestamp(frame, time_sec)

        # Labels
        cv2.putText(frame, "Person 1 (blurred)", (center1[0] - 80, center1[1] + 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
        cv2.putText(frame, "Person 2 (UNBLURRED!)", (center2[0] - 90, center2[1] + 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

        out.write(frame)

    out.release()
    print(f"Generated: {output_path}")


def generate_test_video_3(output_path, face_image_path, duration_sec=8, fps=30):
    """Test 3: Face with mosaic, varying block sizes + gaps."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    face_img, face_mask = load_face_image(face_image_path, (200, 250))
    face_size = (200, 250)
    center = (width // 2, height // 2)
    bbox = get_face_bbox(center, face_size, scale=1.0)

    total_frames = duration_sec * fps

    for frame_idx in range(total_frames):
        bg = create_background(width, height, frame_idx)
        frame = composite_face_onto_background(face_img, face_mask, bg, center, scale=1.0)

        time_sec = frame_idx / fps

        if time_sec < 2.5:
            frame = apply_mosaic(frame, bbox, block_size=8)
        elif 2.5 <= time_sec < 4.5:
            frame = apply_mosaic(frame, bbox, block_size=24)  # Weak mosaic
        elif 4.5 <= time_sec < 5.5:
            pass  # No mosaic
        elif 5.5 <= time_sec < 6.5:
            frame = apply_black_block(frame, bbox)
        else:
            frame = apply_mosaic(frame, bbox, block_size=8)

        # Label
        effects = {
            (0, 2.5): "Mosaic (good)",
            (2.5, 4.5): "Mosaic (WEAK)",
            (4.5, 5.5): "NO EFFECT - exposed!",
            (5.5, 6.5): "Black block",
            (6.5, 8.0): "Mosaic (good)",
        }
        for (s, e), label in effects.items():
            if s <= time_sec < e:
                color = (0, 150, 0) if "good" in label else ((0, 0, 200) if "exposed" in label or "WEAK" in label else (200, 150, 0))
                cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                break

        frame = add_timestamp(frame, time_sec)
        out.write(frame)

    out.release()
    print(f"Generated: {output_path}")


def generate_test_video_4(output_path, face_image_path, duration_sec=8, fps=30):
    """Test 4: Black block shifts/disappears intermittently."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    face_img, face_mask = load_face_image(face_image_path, (180, 220))
    face_size = (180, 220)
    center = (width // 2, height // 2)

    total_frames = duration_sec * fps

    for frame_idx in range(total_frames):
        bg = create_background(width, height, frame_idx)

        # Slight movement
        offset_x = int(10 * np.sin(frame_idx * 0.1))
        offset_y = int(5 * np.cos(frame_idx * 0.15))
        moved_center = (center[0] + offset_x, center[1] + offset_y)

        frame = composite_face_onto_background(face_img, face_mask, bg, moved_center, scale=1.0)
        bbox = get_face_bbox(moved_center, face_size, scale=1.0)

        time_sec = frame_idx / fps

        if time_sec < 1.5:
            frame = apply_black_block(frame, bbox)
        elif 1.5 <= time_sec < 2.2:
            wrong_bbox = [b + 30 for b in bbox]  # Shifted block
            frame = apply_black_block(frame, wrong_bbox)
        elif 2.2 <= time_sec < 3.5:
            frame = apply_black_block(frame, bbox)
        elif 3.5 <= time_sec < 4.0:
            pass  # Block disappears
        elif 4.0 <= time_sec < 6.0:
            frame = apply_black_block(frame, bbox)
        elif 6.0 <= time_sec < 6.5:
            small_bbox = [moved_center[0] - face_size[0]//3, moved_center[1] - face_size[1]//3,
                          moved_center[0] + face_size[0]//3, moved_center[1] + face_size[1]//3]
            frame = apply_black_block(frame, small_bbox)
        else:
            frame = apply_black_block(frame, bbox)

        frame = add_timestamp(frame, time_sec)
        out.write(frame)

    out.release()
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "test_videos")
    os.makedirs(out_dir, exist_ok=True)

    face1 = os.path.join(out_dir, "face1.jpg")
    face2 = os.path.join(out_dir, "face2.jpg")
    face3 = os.path.join(out_dir, "face3.jpg")

    # Check face images exist
    for f in [face1, face2, face3]:
        if not os.path.exists(f):
            print(f"ERROR: Face image not found: {f}")
            print("Please download face images first.")
            sys.exit(1)

    print("Generating test videos with real face images...")

    generate_test_video_1(os.path.join(out_dir, "test_blur_gaps.mp4"), face1, duration_sec=8)
    generate_test_video_2(os.path.join(out_dir, "test_two_people_one_unblurred.mp4"), face1, face2, duration_sec=10)
    generate_test_video_3(os.path.join(out_dir, "test_mosaic_weak.mp4"), face3, duration_sec=8)
    generate_test_video_4(os.path.join(out_dir, "test_black_block_shifts.mp4"), face1, duration_sec=8)

    print(f"\nAll test videos generated in: {out_dir}")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.mp4'):
            size_mb = os.path.getsize(os.path.join(out_dir, f)) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.1f} MB)")
