#!/usr/bin/env python3
"""
Face Anonymizer - Main Application
==================================

A unified face anonymization tool that:
1. Uses YOLO for accurate face detection
2. Provides 4 anonymization functions:
   - Blur faces
   - Pixelate faces  
   - Replace with emoji
   - Synthetic face swap (random or same face)

Usage:
python3 face_anonymizer_main.py --input test/test1.jpg --output test/out.jpg --mode blur
python3 face_anonymizer_main.py --input test/test1.jpg --output test/out.jpg --mode replace --synthetic_dir synthetic_faces --random_faces
"""

from ultralytics import YOLO
import torch
import cv2
import argparse
import os
import numpy as np
import random
from pathlib import Path

# Import anonymization functions
from face_anonymizer import blur_face, pixelate_face, emoji_face, overlay_with_alpha

# Import InSwapper for synthetic face replacement
try:
    import insightface
    from insightface.app import FaceAnalysis
    import onnxruntime as ort
    INSWAPPER_AVAILABLE = True
except ImportError:
    INSWAPPER_AVAILABLE = False
    print("[warning] InSwapper not available. Synthetic face replacement will use classic method.")

class InSwapperBackend:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Load InSwapper model
        model_path = os.path.expanduser('~/.insightface/models/inswapper_128.onnx')
        if os.path.exists(model_path):
            self.swapper = insightface.model_zoo.get_model(model_path)
        else:
            print(f"[warning] InSwapper model not found at {model_path}")
            self.swapper = None
    
    def get_faces(self, image):
        """Detect faces using InsightFace"""
        return self.app.get(image)
    
    def swap_into(self, target_img, target_face, source_face):
        """Swap source face into target image"""
        if self.swapper is None:
            return target_img
        return self.swapper.get(target_img, target_face, source_face, paste_back=True)

def load_images_folder(folder_path):
    """Load all images from a folder"""
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        for img_path in Path(folder_path).glob(ext):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
    return images

def parse_arguments():
    parser = argparse.ArgumentParser(description='Face Anonymizer - YOLO Detection + 4 Anonymization Modes')
    parser.add_argument('--input', required=True, help='Input image or video path')
    parser.add_argument('--output', required=True, help='Output image or video path')
    parser.add_argument('--mode', choices=['blur', 'pixelate', 'emoji', 'replace'], required=True,
                       help='Anonymization mode: blur, pixelate, emoji, or replace (synthetic)')
    parser.add_argument('--weights', default='yolov12l-face.pt', help='YOLO model path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--interactive', action='store_true', 
                       help='Enable interactive face selection (click to select/deselect faces)')
    
    # Mode-specific arguments
    parser.add_argument('--blocks', type=int, default=12, help='Pixelation blocks (for pixelate mode)')
    parser.add_argument('--blur_strength', type=int, default=0, help='Blur kernel size (0=auto)')
    parser.add_argument('--blur_type', choices=['gaussian', 'box', 'median'], default='gaussian', 
                       help='Blur type: gaussian, box, or median (for blur mode)')
    parser.add_argument('--blur_radius', type=float, default=0.0, 
                       help='Gaussian blur radius/sigma (for blur mode)')
    parser.add_argument('--pixelate_method', choices=['nearest', 'linear', 'cubic'], default='nearest',
                       help='Pixelation interpolation method (for pixelate mode)')
    parser.add_argument('--pixelate_quality', type=int, default=50, 
                       help='Pixelation quality percentage 10-100 (for pixelate mode)')
    parser.add_argument('--emoji_path', help='Path to emoji PNG (for emoji mode)')
    parser.add_argument('--synthetic_dir', help='Folder of synthetic faces (for replace mode)')
    parser.add_argument('--random_faces', action='store_true', 
                       help='Use random synthetic faces (default: same face for all)')
    parser.add_argument('--keep_largest', action='store_true', 
                       help='Keep the largest face (main subject) untouched')
    
    return parser.parse_args()

def detect_faces_yolo(image, model, threshold):
    """Detect faces using YOLO"""
    results = model.predict(image, verbose=False)
    faces = []
    
    for result in results:
        bboxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        
        for box, score in zip(bboxes, scores):
            if score >= threshold:
                x1, y1, x2, y2 = map(int, box)
                faces.append((x1, y1, x2, y2, score))
    
    return faces

def mouse_callback(event, x, y, flags, param):
    """Mouse callback for interactive face selection"""
    if event == cv2.EVENT_LBUTTONDOWN:
        param['clicked'] = (x, y)

def interactive_face_selection(image, faces):
    """Interactive face selection using mouse clicks"""
    if not faces:
        return faces
    
    # Create a copy for display
    display_img = image.copy()
    
    # Draw all detected faces
    for i, (x1, y1, x2, y2, score) in enumerate(faces):
        color = (0, 255, 0)  # Green for selected
        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_img, f'Face {i+1}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Setup window and mouse callback
    window_name = 'Face Selection - Click faces to select/deselect, press SPACE when done'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    click_data = {'clicked': None}
    cv2.setMouseCallback(window_name, mouse_callback, click_data)
    
    selected_faces = faces.copy()
    print(f"Detected {len(faces)} faces. Click on faces to select/deselect them.")
    print("Press SPACE when done, ESC to cancel.")
    
    while True:
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(1) & 0xFF
        
        # Check for mouse click
        if click_data['clicked']:
            click_x, click_y = click_data['clicked']
            click_data['clicked'] = None
            
            # Check which face was clicked
            for i, (x1, y1, x2, y2, score) in enumerate(faces):
                if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                    if (x1, y1, x2, y2, score) in selected_faces:
                        # Remove from selection
                        selected_faces.remove((x1, y1, x2, y2, score))
                        color = (0, 0, 255)  # Red for deselected
                        print(f"Face {i+1} deselected")
                    else:
                        # Add to selection
                        selected_faces.append((x1, y1, x2, y2, score))
                        color = (0, 255, 0)  # Green for selected
                        print(f"Face {i+1} selected")
                    
                    # Update display
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_img, f'Face {i+1}', (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    break
        
        # Handle keyboard input
        if key == ord(' '):  # SPACE
            break
        elif key == 27:  # ESC
            selected_faces = []
            break
    
    cv2.destroyAllWindows()
    print(f"Selected {len(selected_faces)} faces for anonymization.")
    return selected_faces

def anonymize_blur(image, faces, blur_strength, blur_type="gaussian", blur_radius=0.0):
    """Anonymize faces with blur using different blur types"""
    result = image.copy()
    for x1, y1, x2, y2, score in faces:
        w, h = x2 - x1, y2 - y1
        roi = result[y1:y2, x1:x2]
        if roi.size == 0:
            continue
            
        # Apply different blur types
        if blur_type == "gaussian":
            if blur_strength <= 0:
                k = max(7, ((max(w, h) // 8) // 2) * 2 + 1)
            else:
                k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
            sigma = blur_radius if blur_radius > 0 else 0
            blurred = cv2.GaussianBlur(roi, (k, k), sigma)
        elif blur_type == "box":
            if blur_strength <= 0:
                k = max(5, max(w, h) // 10)
            else:
                k = blur_strength
            blurred = cv2.blur(roi, (k, k))
        elif blur_type == "median":
            if blur_strength <= 0:
                k = max(5, max(w, h) // 15)
            else:
                k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
            blurred = cv2.medianBlur(roi, k)
        else:
            # Default to Gaussian
            k = max(7, ((max(w, h) // 8) // 2) * 2 + 1)
            blurred = cv2.GaussianBlur(roi, (k, k), 0)
            
        result[y1:y2, x1:x2] = blurred
    return result

def anonymize_pixelate(image, faces, blocks, method="nearest", quality=50):
    """Anonymize faces with pixelation using different methods"""
    result = image.copy()
    for x1, y1, x2, y2, score in faces:
        w, h = x2 - x1, y2 - y1
        roi = result[y1:y2, x1:x2]
        if roi.size == 0:
            continue
            
        # Calculate downsampling based on blocks and quality
        blocks = max(3, blocks)
        quality_factor = quality / 100.0
        
        # Adjust block size based on quality
        effective_blocks = int(blocks * (1.0 - quality_factor * 0.5))
        effective_blocks = max(2, effective_blocks)
        
        down_w = max(1, w // effective_blocks)
        down_h = max(1, h // effective_blocks)
        
        # Choose interpolation method
        if method == "nearest":
            down_interp = cv2.INTER_LINEAR
            up_interp = cv2.INTER_NEAREST
        elif method == "linear":
            down_interp = cv2.INTER_LINEAR
            up_interp = cv2.INTER_LINEAR
        elif method == "cubic":
            down_interp = cv2.INTER_CUBIC
            up_interp = cv2.INTER_CUBIC
        else:
            down_interp = cv2.INTER_LINEAR
            up_interp = cv2.INTER_NEAREST
        
        # Apply pixelation
        small = cv2.resize(roi, (down_w, down_h), interpolation=down_interp)
        pix = cv2.resize(small, (w, h), interpolation=up_interp)
        
        result[y1:y2, x1:x2] = pix
    return result

def anonymize_emoji(image, faces, emoji_path):
    """Anonymize faces with emoji"""
    if not emoji_path or not os.path.exists(emoji_path):
        print("[error] Emoji path not found:", emoji_path)
        return image
    
    emoji_img = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    if emoji_img is None:
        print("[error] Failed to load emoji image")
        return image
    
    result = image.copy()
    for x1, y1, x2, y2, score in faces:
        w, h = x2 - x1, y2 - y1
        emoji_face(result, x1, y1, w, h, emoji_img)
    return result

def anonymize_synthetic(image, faces, synthetic_dir, random_faces=False, keep_largest=False):
    """Anonymize faces with synthetic face replacement"""
    if not synthetic_dir or not os.path.exists(synthetic_dir):
        print("[error] Synthetic directory not found:", synthetic_dir)
        return image
    
    # Use InSwapper if available, otherwise fall back to classic method
    if INSWAPPER_AVAILABLE:
        return anonymize_synthetic_inswapper(image, faces, synthetic_dir, random_faces, keep_largest)
    else:
        return anonymize_synthetic_classic(image, faces, synthetic_dir, random_faces, keep_largest)

def anonymize_synthetic_inswapper(image, faces, synthetic_dir, random_faces=False, keep_largest=False):
    """High-quality synthetic face replacement using InSwapper"""
    try:
        backend = InSwapperBackend()
        sources = load_images_folder(synthetic_dir)
        
        if not sources:
            print("[error] No synthetic faces found in directory")
            return image
        
        # Determine which faces to process
        if keep_largest and faces:
            areas = [(x2-x1)*(y2-y1) for x1, y1, x2, y2, score in faces]
            largest_idx = np.argmax(areas)
        else:
            largest_idx = None
        
        # Get InsightFace detections
        insight_faces = backend.get_faces(image)
        if not insight_faces:
            print("[warning] No faces detected by InsightFace, using classic method")
            return anonymize_synthetic_classic(image, faces, synthetic_dir, random_faces, keep_largest)
        
        # Match YOLO boxes with InsightFace faces
        matched_faces = []
        for i, (x1, y1, x2, y2, score) in enumerate(faces):
            if keep_largest and i == largest_idx:
                continue
                
            # Find best matching InsightFace detection
            best_match = None
            best_iou = 0
            for face in insight_faces:
                fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                intersection_x1 = max(x1, fx1)
                intersection_y1 = max(y1, fy1)
                intersection_x2 = min(x2, fx2)
                intersection_y2 = min(y2, fy2)
                
                if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                    union_area = (x2 - x1) * (y2 - y1) + (fx2 - fx1) * (fy2 - fy1) - intersection_area
                    iou = intersection_area / union_area if union_area > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = face
            
            if best_match and best_iou > 0.3:
                matched_faces.append((best_match, (x1, y1, x2, y2, score)))
        
        if not matched_faces:
            return image
        
        # Prepare source faces
        src_faces_cached = []
        for src_img in sources:
            src_face_list = backend.get_faces(src_img)
            if src_face_list:
                src_face = max(src_face_list, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                src_faces_cached.append((src_img, src_face))
        
        if not src_faces_cached:
            return image
        
        # Apply InSwapper
        result = image.copy()
        chosen = random.choice(src_faces_cached) if not random_faces else None
        
        for face, (x1, y1, x2, y2, score) in matched_faces:
            if not random_faces:
                src_img, src_face = chosen
            else:
                src_img, src_face = random.choice(src_faces_cached)
            
            result = backend.swap_into(result, face, src_face)
        
        return result
        
    except Exception as e:
        print(f"[warning] InSwapper failed: {e}. Falling back to classic method.")
        return anonymize_synthetic_classic(image, faces, synthetic_dir, random_faces, keep_largest)

def anonymize_synthetic_classic(image, faces, synthetic_dir, random_faces=False, keep_largest=False):
    """Classic synthetic face replacement using OpenCV"""
    # Load synthetic faces
    synthetic_faces = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        synthetic_faces.extend(Path(synthetic_dir).glob(ext))
    
    if not synthetic_faces:
        print("[error] No synthetic faces found")
        return image
    
    # Determine which faces to process
    if keep_largest and faces:
        areas = [(x2-x1)*(y2-y1) for x1, y1, x2, y2, score in faces]
        largest_idx = np.argmax(areas)
    else:
        largest_idx = None
    
    result = image.copy()
    chosen_face = None
    
    for i, (x1, y1, x2, y2, score) in enumerate(faces):
        if keep_largest and i == largest_idx:
            continue
        
        # Choose synthetic face
        if random_faces or chosen_face is None:
            face_path = random.choice(synthetic_faces)
            chosen_face = cv2.imread(str(face_path))
        
        if chosen_face is not None:
            w, h = x2 - x1, y2 - y1
            # Simple face replacement (resize and paste)
            face_resized = cv2.resize(chosen_face, (w, h))
            result[y1:y2, x1:x2] = face_resized
    
    return result

def main():
    args = parse_arguments()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"[error] Input file not found: {args.input}")
        return
    
    # Load YOLO model
    try:
        model = YOLO(args.weights)
    except Exception as e:
        print(f"[error] Failed to load YOLO model: {e}")
        return
    
    # Check if it's a video by file extension
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    file_ext = os.path.splitext(args.input)[1].lower()
    
    if file_ext in video_extensions:
        process_video(args, model)
    else:
        process_image(args, model)

def process_image(args, model):
    """Process a single image"""
    # Load image
    image = cv2.imread(args.input)
    if image is None:
        print(f"[error] Failed to load image: {args.input}")
        return
    
    # Detect faces
    print(f"Detecting faces in {args.input}...")
    faces = detect_faces_yolo(image, model, args.threshold)
    print(f"Detected {len(faces)} faces")
    
    if not faces:
        print("No faces detected. Saving original image.")
        cv2.imwrite(args.output, image)
        return
    
    # Interactive face selection if enabled
    if args.interactive:
        print("Starting interactive face selection...")
        faces = interactive_face_selection(image, faces)
        if not faces:
            print("No faces selected. Saving original image.")
            cv2.imwrite(args.output, image)
            return
    
    # Apply anonymization
    print(f"Applying {args.mode} anonymization...")
    if args.mode == 'blur':
        result = anonymize_blur(image, faces, args.blur_strength, 
                               getattr(args, 'blur_type', 'gaussian'), 
                               getattr(args, 'blur_radius', 0.0))
    elif args.mode == 'pixelate':
        result = anonymize_pixelate(image, faces, args.blocks, 
                                   getattr(args, 'pixelate_method', 'nearest'), 
                                   getattr(args, 'pixelate_quality', 50))
    elif args.mode == 'emoji':
        result = anonymize_emoji(image, faces, args.emoji_path)
    elif args.mode == 'replace':
        result = anonymize_synthetic(image, faces, args.synthetic_dir, args.random_faces, args.keep_largest)
    
    # Save result
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, result)
    print(f"Saved anonymized image: {args.output}")

def process_video(args, model):
    """Process a video"""
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[error] Failed to open video: {args.input}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print(f"Processing video: {total_frames} frames at {fps} fps")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = detect_faces_yolo(frame, model, args.threshold)
        
        # Apply anonymization
        if faces:
            if args.mode == 'blur':
                frame = anonymize_blur(frame, faces, args.blur_strength)
            elif args.mode == 'pixelate':
                frame = anonymize_pixelate(frame, faces, args.blocks)
            elif args.mode == 'emoji':
                frame = anonymize_emoji(frame, faces, args.emoji_path)
            elif args.mode == 'replace':
                frame = anonymize_synthetic(frame, faces, args.synthetic_dir, args.random_faces, args.keep_largest)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    out.release()
    print(f"Saved anonymized video: {args.output}")

if __name__ == '__main__':
    main()
