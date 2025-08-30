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

### 2. Download Required Models
Download the PII detection models from: [https://drive.google.com/drive/folders/1fxljdNcruwfbLTIBIjWU-dLWR2z-Gn3U?usp=sharing](https://drive.google.com/drive/folders/1fxljdNcruwfbLTIBIjWU-dLWR2z-Gn3U?usp=sharing)

Available models:
- `pii_deberta_base_v1` - DeBERTa-based PII detection model
- `pii_ettin_encoder_1b_v1` - Ettin 1B encoder model (recommended)
- `pii_ettin_encoder_400m_v1` - Ettin 400M encoder model
- `pii_modernbert_base_v1` - ModernBERT-based PII detection model

### 3.set up yolo

MacOS
```bash
curl -L -o yolov12l-face.pt "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12l-face.pt"

mkdir -p ~/.insightface/models

# Download InSwapper model
gdown "https://drive.google.com/uc?id=1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF" \
  -O ~/.insightface/models/inswapper_128.onnx
```

Windows:
```bash
Invoke-WebRequest "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12l-face.pt" -OutFile "yolov12l-face.pt"

mkdir .insightface\models

gdown "https://drive.google.com/uc?id=1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF" `
  -O ".insightface\models\inswapper_128.onnx"
```

### 4. Launch the Application

MacOS
```bash
python3 face_anonymizer_gui_modern.py
```

Windows:
```bash
python face_anonymizer_gui_modern.py
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
