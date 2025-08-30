# Face Anonymizer - YOLO Detection + 4 Anonymization Modes

A unified face anonymization tool that uses **YOLO for accurate face detection** and provides **4 anonymization functions**:

1. **Blur faces** - Gaussian blur with adaptive kernel
2. **Pixelate faces** - Block-based pixelation
3. **Replace with emoji** - Overlay PNG emoji
4. **Synthetic face swap** - High-quality face replacement (random or same face)

## 🚀 Features

- **YOLO Detection**: Accurate face detection using YOLOv12l-face
- **4 Anonymization Modes**: blur, pixelate, emoji, replace
- **Interactive Selection**: Click to select/deselect faces for anonymization
- **Synthetic Face Options**: Random faces or same face for all
- **Bystander Protection**: Keep the largest face (main subject) untouched
- **Video Support**: Works with both images and videos
- **High-Quality**: InSwapper integration for realistic face replacement

## 📦 Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
python3 face_anonymizer_gui_modern.py
```

The GUI provides:
- Easy file selection for input/output
- Visual mode selection (blur, pixelate, emoji, replace)
- Interactive sliders for parameters
- **Advanced blur options**: Choose blur type (Gaussian, Box, Median) with radius control
- **Advanced pixelation**: Select interpolation method (Nearest, Linear, Cubic) with quality control
- **Emoji selection**: Choose from 5 built-in emojis with live preview
- **Interactive face selection**: Click faces to select/deselect them
- Progress feedback and status updates
- No command line knowledge required!
