#!/usr/bin/env python3
"""
Face Anonymizer — Blur / Pixelate / Emoji / Replace (Synthetic, Natural Blend)
=============================================================================

Modes
-----
- blur       : Gaussian blur faces
- pixelate   : Pixelate faces
- emoji      : Cover faces with a PNG (alpha supported)
- replace    : Replace faces with synthetic faces from a folder, with color/lighting
               matching and smooth Poisson blending to look natural.

Key flags
---------
--keep_largest : Treat largest face as the main subject and leave it untouched
--all_faces    : Override keep_largest; anonymize everyone
--scale_pad    : How much to expand the destination face region before replacement (1.1–1.4 is good)
--feather      : Mask softness for blending (odd int, e.g., 41)
--clone_mode   : Poisson blend mode: 'normal' (photoreal) or 'mixed' (preserve edges)

Examples
--------
# Blur
python3 face_anonymizer.py --input input.jpg --output out.jpg --mode blur

# Pixelate
python3 face_anonymizer.py --input input.jpg --output out.jpg --mode pixelate --blocks 10

# Emoji
python3 face_anonymizer.py --input input.jpg --output out.jpg --mode emoji --emoji_path smile.png

# Replace bystanders (photo)
python3 face_anonymizer.py --input input.jpg --output out.jpg --mode replace  \
  --synthetic_dir ./synthetic_faces --keep_largest --scale_pad 1.25 --feather 41 --clone_mode normal

# Replace in video
python3 face_anonymizer.py --input input.mp4 --output out.mp4 --mode replace \
  --synthetic_dir ./synthetic_faces --keep_largest

"""

import argparse
import os
import sys
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

# Optional tqdm (progress bar). Falls back gracefully if missing.
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # ModuleNotFoundError, etc.
    class _TqdmShim:  # minimal stand-in
        def __init__(self, total=None, desc=None):
            self.total = total
            self.n = 0
        def update(self, n=1):
            self.n += n
        def close(self):
            pass
    def tqdm(*args, **kwargs):  # type: ignore
        return _TqdmShim(*args, **kwargs)


# ----------------------------- Face detection ----------------------------- #

def load_cascade(kind: str) -> cv2.CascadeClassifier:
    """Load an OpenCV Haar cascade by short name."""
    name_map = {
        "default": "haarcascade_frontalface_default.xml",
        "alt2": "haarcascade_frontalface_alt2.xml",
        "profile": "haarcascade_profileface.xml",
    }
    fname = name_map.get(kind, name_map["default"])
    cascade_path = os.path.join(cv2.data.haarcascades, fname)
    if not os.path.exists(cascade_path):
        print(f"[error] Could not find Haar cascade at: {cascade_path}", file=sys.stderr)
        sys.exit(1)

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print("[error] Failed to load face cascade.", file=sys.stderr)
        sys.exit(1)
    return cascade


def detect_faces(
    frame_bgr: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    downsamp: float = 1.0,
    min_size_px: int = 40,
    scaleFactor: float = 1.2,
    minNeighbors: int = 5,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces and return boxes in ORIGINAL image coordinates.
    downsamp > 1.0 means we detect on a smaller, faster image (width/downsamp).
    """
    h, w = frame_bgr.shape[:2]
    if downsamp > 1.0:
        small_w = max(1, int(w / downsamp))
        small_h = max(1, int(h / downsamp))
        frame_small = cv2.resize(frame_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        scale_back = downsamp
    else:
        frame_small = frame_bgr
        scale_back = 1.0

    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    min_size_small = max(20, int(min_size_px / scale_back))
    faces_small = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=(min_size_small, min_size_small),
    )

    faces = []
    for (x, y, fw, fh) in faces_small:
        X = int(x * scale_back)
        Y = int(y * scale_back)
        W = int(fw * scale_back)
        H = int(fh * scale_back)
        # clamp to image bounds
        X = max(0, min(X, w - 1))
        Y = max(0, min(Y, h - 1))
        W = max(1, min(W, w - X))
        H = max(1, min(H, h - Y))
        faces.append((X, Y, W, H))

    return faces


def largest_face_index(boxes: List[Tuple[int, int, int, int]]) -> Optional[int]:
    if not boxes:
        return None
    areas = [w * h for (_, _, w, h) in boxes]
    return int(np.argmax(areas))


# ----------------------------- Anonymization ops ----------------------------- #

def blur_face(frame: np.ndarray, x: int, y: int, w: int, h: int, blur_strength: int = 0) -> None:
    """In-place Gaussian blur. If blur_strength == 0, choose an adaptive odd kernel."""
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return
    if blur_strength <= 0:
        # kernel size roughly proportional to face size, ensure odd
        k = max(7, ((max(w, h) // 8) // 2) * 2 + 1)
    else:
        k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    frame[y:y+h, x:x+w] = blurred


def pixelate_face(frame: np.ndarray, x: int, y: int, w: int, h: int, blocks: int = 12) -> None:
    """In-place pixelation by downscaling then upscaling with NEAREST."""
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return
    blocks = max(3, blocks)
    down_w = max(1, w // blocks)
    down_h = max(1, h // blocks)
    small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y:y+h, x:x+w] = pix


def overlay_with_alpha(bg: np.ndarray, fg: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """Overlay fg (BGR or BGRA) onto bg at (x,y) with size (w,h), alpha-respecting if present."""
    H, W = bg.shape[:2]
    if w <= 0 or h <= 0 or x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return

    fg_resized = cv2.resize(fg, (w, h), interpolation=cv2.INTER_AREA)

    if fg_resized.shape[2] == 3:
        alpha = np.ones((h, w), dtype=np.float32)
        fg_rgb = fg_resized.astype(np.float32)
    else:
        alpha = fg_resized[:, :, 3].astype(np.float32) / 255.0
        fg_rgb = fg_resized[:, :, :3].astype(np.float32)

    # clamp ROI to bg
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return

    fg_x0, fg_y0 = x0 - x, y0 - y
    fg_x1, fg_y1 = fg_x0 + (x1 - x0), fg_y0 + (y1 - y0)

    roi = bg[y0:y1, x0:x1].astype(np.float32)
    fg_crop = fg_rgb[fg_y0:fg_y1, fg_x0:fg_x1]
    a = alpha[fg_y0:fg_y1, fg_x0:fg_x1][..., None]

    blended = a * fg_crop + (1 - a) * roi
    bg[y0:y1, x0:x1] = blended.astype(np.uint8)


def emoji_face(frame: np.ndarray, x: int, y: int, w: int, h: int, emoji_img: np.ndarray) -> None:
    """Overlay an emoji PNG over the face with padding to ensure full coverage."""
    pad = int(0.15 * max(w, h))
    overlay_with_alpha(frame, emoji_img, x - pad, y - pad, w + 2 * pad, h + 2 * pad)


# ---------------------- Natural-looking synthetic replace ------------------- #

def lab_color_transfer(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """Reinhard-style color transfer (match mean/std in L*a*b*)."""
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    s_mu, s_sigma = cv2.meanStdDev(src)
    r_mu, r_sigma = cv2.meanStdDev(ref)
    s_sigma = np.maximum(s_sigma, 1e-6)
    r_sigma = np.maximum(r_sigma, 1e-6)

    out = (src - s_mu.reshape(1,1,3)) * (r_sigma.reshape(1,1,3)/s_sigma.reshape(1,1,3)) + r_mu.reshape(1,1,3)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def replace_face_natural(frame: np.ndarray,
                         x: int, y: int, w: int, h: int,
                         synth_pool: List[np.ndarray],
                         scale_pad: float = 1.2,
                         feather: int = 41,
                         clone_mode: str = "normal") -> None:
    """Improved replacement: color-match + distance-weighted mask + selectable clone mode."""
    if not synth_pool:
        return
    H, W = frame.shape[:2]
    src = random.choice(synth_pool)
    if src is None or src.size == 0:
        return

    cx, cy = x + w // 2, y + h // 2
    tw = int(w * scale_pad)
    th = int(h * scale_pad)
    tx = max(0, cx - tw // 2)
    ty = max(0, cy - th // 2)
    tw = min(W - tx, tw)
    th = min(H - ty, th)
    if tw <= 0 or th <= 0:
        return

    src_bgr = src[:, :, :3] if src.shape[2] == 4 else src
    warped = cv2.resize(src_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)

    # Color transfer to local target ROI
    target_roi = frame[ty:ty+th, tx:tx+tw]
    try:
        warped = lab_color_transfer(warped, target_roi)
    except Exception:
        pass

    # Soft elliptical mask + distance transform for smoother roll-off
    mask = np.zeros((H, W), dtype=np.uint8)
    axes = (int(tw * 0.45), int(th * 0.60))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    if feather > 0:
        k = feather | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    if dist.max() > 0:
        dist = (dist / dist.max() * 255).astype(np.uint8)
        mask = cv2.max(mask, dist)

    # Compose on a canvas
    canvas = frame.copy()
    canvas[ty:ty+th, tx:tx+tw] = warped

    clone_flag = cv2.NORMAL_CLONE if clone_mode == "normal" else cv2.MIXED_CLONE
    try:
        blended = cv2.seamlessClone(canvas, frame, mask, (cx, cy), clone_flag)
        frame[:, :] = blended
    except cv2.error:
        m = (mask.astype(np.float32) / 255.0)[..., None]
        frame[:, :] = (canvas.astype(np.float32) * m + frame.astype(np.float32) * (1 - m)).astype(np.uint8)


# ----------------------------- Processing helpers ----------------------------- #

def process_image(
    img_bgr: np.ndarray,
    cascade: cv2.CascadeClassifier,
    mode: str,
    emoji_img: np.ndarray = None,
    synth_pool: List[np.ndarray] = None,
    keep_largest: bool = False,
    downsamp: float = 1.0,
    min_size_px: int = 40,
    blur_strength: int = 0,
    blocks: int = 12,
    scaleFactor: float = 1.2,
    minNeighbors: int = 5,
    scale_pad: float = 1.2,
    feather: int = 41,
    clone_mode: str = "normal",
) -> np.ndarray:
    """Detect faces, apply the chosen anonymization, and return the processed image."""
    boxes = detect_faces(
        img_bgr,
        cascade,
        downsamp=downsamp,
        min_size_px=min_size_px,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
    )

    skip_idx = largest_face_index(boxes) if keep_largest else None

    for idx, (x, y, w, h) in enumerate(boxes):
        if keep_largest and idx == skip_idx:
            continue
        if mode == "blur":
            blur_face(img_bgr, x, y, w, h, blur_strength=blur_strength)
        elif mode == "pixelate":
            pixelate_face(img_bgr, x, y, w, h, blocks=blocks)
        elif mode == "emoji":
            if emoji_img is None:
                continue
            emoji_face(img_bgr, x, y, w, h, emoji_img)
        elif mode == "replace":
            replace_face_natural(
                img_bgr, x, y, w, h,
                synth_pool or [],
                scale_pad=scale_pad,
                feather=feather,
                clone_mode=clone_mode,
            )

    return img_bgr


def read_image_maybe_video(path: str) -> Tuple[Optional[np.ndarray], Optional[cv2.VideoCapture]]:
    """Try to read an image first. If None, return a VideoCapture (for completeness)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img, None
    cap = cv2.VideoCapture(path)
    return (None, cap) if cap.isOpened() else (None, None)


def load_synthetic_pool(folder: Optional[str]) -> List[np.ndarray]:
    if not folder:
        return []
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        print(f"[error] Synthetic dir not found: {folder}", file=sys.stderr)
        sys.exit(1)
    imgs: List[np.ndarray] = []
    for f in sorted(p.iterdir()):
        if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] in (3, 4):
            imgs.append(img)
    if not imgs:
        print("[error] No usable images in synthetic dir (expect PNG/JPG with one face).", file=sys.stderr)
        sys.exit(1)
    return imgs


# ----------------------------- CLI ----------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Safe face anonymizer (images & videos). Adds synthetic face replacement with color matching.")
    p.add_argument("--input", required=True, help="Path to input image or video.")
    p.add_argument("--output", required=True, help="Path to output image or video.")
    p.add_argument("--mode", choices=["blur", "pixelate", "emoji", "replace"], default="blur",
                   help="Type of anonymization to apply.")
    p.add_argument("--emoji_path", help="PNG (with transparency) for --mode emoji.")
    p.add_argument("--blocks", type=int, default=12,
                   help="Pixelate blocks across face width (bigger squares = smaller number).")
    p.add_argument("--blur_strength", type=int, default=0,
                   help="Odd kernel size for blur. 0 = auto based on face size.")
    p.add_argument("--cascade", choices=["default", "alt2", "profile"], default="default",
                   help="Choose Haar cascade. 'alt2' can be more robust for some photos.")
    p.add_argument("--min_size_px", type=int, default=40,
                   help="Minimum detected face size in ORIGINAL pixels.")
    p.add_argument("--downsamp", type=float, default=1.0,
                   help=">1.0 downsamples for faster detection (e.g., 1.5 or 2.0).")
    p.add_argument("--scaleFactor", type=float, default=1.2,
                   help="Detector scaleFactor (1.05–1.3). Smaller = slower, possibly more accurate.")
    p.add_argument("--minNeighbors", type=int, default=5,
                   help="Detector minNeighbors. Larger = fewer detections, less false positives.")
    # New for replace mode / bystanders
    p.add_argument("--synthetic_dir", help="Folder of synthetic faces for --mode replace.")
    p.add_argument("--keep_largest", action="store_true", help="Skip the largest face (treat others as bystanders).")
    p.add_argument("--all_faces", action="store_true", help="Anonymize all faces (overrides --keep_largest).")
    p.add_argument("--scale_pad", type=float, default=1.2, help="Scale target box before replace (1.1–1.4).")
    p.add_argument("--feather", type=int, default=41,
                   help="Mask feather (odd int). Larger = smoother blend.")
    p.add_argument("--clone_mode", choices=["normal", "mixed"], default="normal",
                   help="Poisson clone mode: normal = photoreal skin; mixed = preserve edges.")
    return p


def main():
    args = build_parser().parse_args()

    # Validate input path
    if not os.path.exists(args.input):
        print(f"[error] Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load cascade
    cascade = load_cascade(args.cascade)

    # Load emoji if needed
    emoji_img = None
    if args.mode == "emoji":
        if not args.emoji_path or not os.path.exists(args.emoji_path):
            print("[error] Please provide a valid --emoji_path PNG for emoji mode.", file=sys.stderr)
            sys.exit(1)
        emoji_img = cv2.imread(args.emoji_path, cv2.IMREAD_UNCHANGED)
        if emoji_img is None or emoji_img.ndim != 3 or emoji_img.shape[2] not in (3, 4):
            print("[error] Failed to read emoji image (expect PNG with 3 or 4 channels).", file=sys.stderr)
            sys.exit(1)

    # Load synthetic pool if needed
    synth_pool: List[np.ndarray] = []
    if args.mode == "replace":
        if not args.synthetic_dir:
            print("[error] --synthetic_dir is required for --mode replace.", file=sys.stderr)
            sys.exit(1)
        synth_pool = load_synthetic_pool(args.synthetic_dir)

    keep_largest = False if args.all_faces else args.keep_largest

    # Try as image first
    img, cap = read_image_maybe_video(args.input)
    if img is not None:
        out = process_image(
            img,
            cascade,
            mode=args.mode,
            emoji_img=emoji_img,
            synth_pool=synth_pool,
            keep_largest=keep_largest,
            downsamp=max(1.0, args.downsamp),
            min_size_px=max(10, args.min_size_px),
            blur_strength=args.blur_strength,
            blocks=args.blocks,
            scaleFactor=args.scaleFactor,
            minNeighbors=args.minNeighbors,
            scale_pad=args.scale_pad,
            feather=args.feather,
            clone_mode=args.clone_mode,
        )
        ok = cv2.imwrite(args.output, out)
        if not ok:
            print("[error] Failed to write output image.", file=sys.stderr)
            sys.exit(1)
        print(f"[ok] Saved anonymized image → {args.output}")
        return

    # Fallback: process as video
    if cap is None:
        print("[error] Could not open input as image or video.", file=sys.stderr)
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if args.output.lower().endswith(".mp4") else "XVID"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    if not writer.isOpened():
        print("[error] Failed to open video writer for output.", file=sys.stderr)
        cap.release()
        sys.exit(1)

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), desc="Processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_image(
            frame,
            cascade,
            mode=args.mode,
            emoji_img=emoji_img,
            synth_pool=synth_pool,
            keep_largest=keep_largest,
            downsamp=max(1.0, args.downsamp),
            min_size_px=max(10, args.min_size_px),
            blur_strength=args.blur_strength,
            blocks=args.blocks,
            scaleFactor=args.scaleFactor,
            minNeighbors=args.minNeighbors,
            scale_pad=args.scale_pad,
            feather=args.feather,
            clone_mode=args.clone_mode,
        )
        writer.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print(f"[ok] Saved anonymized video → {args.output}")


if __name__ == "__main__":
    main()
