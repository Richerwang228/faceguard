"""
Generate a high-speed motion test video using real face images.
Simulates fast left-right moving faces to test detector/tracker under motion.
"""

import cv2
import numpy as np
import os
import sys


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


def apply_black_block(image, bbox):
    """Apply solid black block."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return image


def generate_highspeed_motion_video(output_path, face_image_path, duration_sec=5, fps=30):
    """
    Test video: Real face moving rapidly left-right across the frame.
    Simulates: person running/walking quickly past camera.
    Includes periods of blur protection and exposed periods.
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load face image
    face_img = cv2.imread(face_image_path)
    if face_img is None:
        raise ValueError(f"Cannot load face image: {face_image_path}")

    # Resize face to reasonable size
    face_size = 120
    face_img = cv2.resize(face_img, (face_size * 2, face_size * 2))
    fh, fw = face_img.shape[:2]

    total_frames = duration_sec * fps

    for frame_idx in range(total_frames):
        frame = np.full((height, width, 3), (240, 240, 245), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (width, height // 2), (200, 210, 220), -1)
        cv2.rectangle(frame, (0, height // 2), (width, height), (180, 170, 160), -1)

        time_sec = frame_idx / fps

        # Fast horizontal oscillation
        t = time_sec
        x_pos = int(width * 0.5 + width * 0.35 * np.sin(t * 2 * np.pi * 0.6))
        y_pos = int(height * 0.5 + 15 * np.sin(t * 2 * np.pi * 1.2))

        # Compute paste position (centered)
        x1 = x_pos - fw // 2
        y1 = y_pos - fh // 2
        x2 = x1 + fw
        y2 = y1 + fh

        # Clip to frame bounds
        px1, py1 = max(0, x1), max(0, y1)
        px2, py2 = min(width, x2), min(height, y2)

        # Source region from face image
        sx1 = px1 - x1
        sy1 = py1 - y1
        sx2 = sx1 + (px2 - px1)
        sy2 = sy1 + (py2 - py1)

        if px2 > px1 and py2 > py1 and sx2 > sx1 and sy2 > sy1:
            frame[py1:py2, px1:px2] = face_img[sy1:sy2, sx1:sx2]

        bbox = [x1, y1, x2, y2]

        # Apply effects: blur for first 2s, exposed 2-3s, black block 3-5s
        if time_sec < 2.0:
            frame = apply_blur(frame, bbox, strength=51)
        elif 2.0 <= time_sec < 3.0:
            pass  # EXPOSED
        else:
            frame = apply_black_block(frame, bbox)

        # Speed indicator
        speed = abs(np.cos(t * 2 * np.pi * 0.6) * width * 0.35 * 2 * np.pi * 0.6)
        cv2.putText(frame, f"Speed: {speed:.0f} px/s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)

        if time_sec < 2.0:
            label, color = "BLURRED", (0, 150, 0)
        elif time_sec < 3.0:
            label, color = "EXPOSED!", (0, 0, 200)
        else:
            label, color = "BLACK BLOCK", (0, 150, 0)
        cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ts = f"{int(time_sec // 60):02d}:{int(time_sec % 60):02d}"
        cv2.putText(frame, ts, (width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)

        out.write(frame)

    out.release()
    max_speed = int(width * 0.35 * 2 * np.pi * 0.6)
    print(f"Generated: {output_path}")
    print(f"  Duration: {duration_sec}s, FPS: {fps}, Frames: {total_frames}")
    print(f"  Max speed: ~{max_speed} px/s")
    return output_path


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "test_videos")
    os.makedirs(out_dir, exist_ok=True)

    face_img = os.path.join(out_dir, "face1.jpg")
    if not os.path.exists(face_img):
        print(f"Face image not found: {face_img}")
        print("Using fallback: generating with synthetic face...")
        sys.exit(1)

    generate_highspeed_motion_video(
        os.path.join(out_dir, "test_highspeed_motion.mp4"),
        face_img,
        duration_sec=5,
        fps=30,
    )
