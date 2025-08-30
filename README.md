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

### 2. Download YOLO Face Model
```bash
curl -L -o yolov12l-face.pt "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12l-face.pt"
```

### 3. Download InSwapper Model (for high-quality replacement)
```bash
# Create directory
mkdir -p ~/.insightface/models

# Download InSwapper model
gdown "https://drive.google.com/uc?id=1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF" \
  -O ~/.insightface/models/inswapper_128.onnx
```

## 🎯 Usage

### Option 1: Simple GUI (Recommended)

**Launch the graphical interface:**
```bash
# Option 1: Direct launch
python3 face_anonymizer_gui.py

# Option 2: Using launcher script
python3 run_gui.py

# Option 3: Make executable and run directly
chmod +x run_gui.py
./run_gui.py
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

### Option 2: Command Line: `face_anonymizer_main.py`

This is the **unified main script** that handles all 4 anonymization modes with YOLO detection.

#### 1. Blur Faces
```bash
# Basic blur
python3 face_anonymizer_main.py --input test/test1.jpg --output test/out_blur.jpg --mode blur

# Advanced blur with custom parameters
python3 face_anonymizer_main.py --input test/test1.jpg --output test/out_blur.jpg --mode blur \
  --blur_type gaussian --blur_strength 15 --blur_radius 2.0

# Box blur
python3 face_anonymizer_main.py --input test/test1.jpg --output test/out_blur.jpg --mode blur \
  --blur_type box --blur_strength 20

# Median blur
python3 face_anonymizer_main.py --input test/test1.jpg --output test/out_blur.jpg --mode blur \
  --blur_type median --blur_strength 11
```

#### 2. Pixelate Faces
```bash
# Basic pixelation
python3 face_anonymizer_main.py --input test/test1.jpg --output test/out_pixelate.jpg --mode pixelate --blocks 10

# Advanced pixelation with custom parameters
python3 face_anonymizer_main.py --input test/test1.jpg --output test/out_pixelate.jpg --mode pixelate \
  --blocks 8 --pixelate_method linear --pixelate_quality 75

# High-quality cubic pixelation
python3 face_anonymizer_main.py --input test/test1.jpg --output test/out_pixelate.jpg --mode pixelate \
  --blocks 6 --pixelate_method cubic --pixelate_quality 90
```

#### 3. Replace with Emoji
```bash
python3 face_anonymizer_main.py --input test/test1.jpg --output test/out_emoji.jpg --mode emoji --emoji_path cat.png
```

**Available Emojis**: The GUI includes 5 built-in emoji options:
- **Baozou** - Classic meme face
- **Cat** - Cute cat emoji  
- **Doge** - Popular doge meme
- **Shrek** - Shrek character
- **Yao** - Yao Ming meme face

**Note**: In the GUI, you can select from these emojis using radio buttons with live preview. For command line, specify the emoji file path.

#### 4. Synthetic Face Swap (High-Quality InSwapper)

**Same synthetic face for all:**
```bash
python3 face_anonymizer_main.py --input test/test1.jpg --output test/out_replace.jpg --mode replace --synthetic_dir synthetic_faces
```

**Random synthetic faces:**
```bash
python3 face_anonymizer_main.py --input test/test1.jpg --output test/out_replace.jpg --mode replace --synthetic_dir synthetic_faces --random_faces
```

**Keep main subject untouched:**
```bash
python3 face_anonymizer_main.py --input test/test1.jpg --output test/out_replace.jpg --mode replace --synthetic_dir synthetic_faces --keep_largest
```

### Interactive Face Selection

**Enable interactive mode to select which faces to anonymize:**
```bash
python3 face_anonymizer_main.py --input test/test1.jpg --output test/out_interactive.jpg --mode blur --interactive
```

**Interactive synthetic face replacement:**
```bash
python3 face_anonymizer_main.py --input test/test1.jpg --output test/out_interactive.jpg --mode replace --synthetic_dir synthetic_faces --interactive
```

**In the GUI:**
1. Check "Interactive Face Selection" option
2. Process your image
3. A window will open showing detected faces with green boxes
4. Click on faces to select (green) or deselect (red) them
5. Press SPACE when done, or ESC to cancel
6. Only selected faces will be anonymized

**Note:** The interactive window runs in the main thread to avoid OpenCV compatibility issues.

### Advanced Options

- `--threshold`: Detection confidence threshold (default: 0.5)
- `--interactive`: Enable interactive face selection (click to select/deselect)
- `--blur_strength`: Blur kernel size (0=auto)
- `--blur_type`: Blur type - gaussian, box, median (default: gaussian)
- `--blur_radius`: Gaussian blur radius/sigma (default: 0.0)
- `--blocks`: Pixelation blocks (default: 12)
- `--pixelate_method`: Pixelation method - nearest, linear, cubic (default: nearest)
- `--pixelate_quality`: Pixelation quality percentage 10-100 (default: 50)
- `--random_faces`: Use random synthetic faces (default: same face)
- `--keep_largest`: Keep the largest face (main subject) untouched

## 📁 Project Structure

```
tiktoktechjam/
├── face_anonymizer_gui.py   # 🎯 GUI: Simple graphical interface
├── run_gui.py              # Easy launcher script
├── face_anonymizer_main.py  # Command line application
├── face_anonymizer.py       # Classic OpenCV-based functions
├── yolov12l-face.pt        # YOLO face detection model
├── emoji/                  # Built-in emoji collection
│   ├── Baozou.png         # Classic meme face
│   ├── Cat.png            # Cute cat emoji
│   ├── Doge.png           # Popular doge meme
│   ├── Shrek.png          # Shrek character
│   └── Yao.png            # Yao Ming meme face
├── synthetic_faces/        # Folder for synthetic face images
├── test/                   # Test images
├── requirements.txt        # All dependencies
└── README.md              # This file
```

## 🔄 How It Works

### Flow Diagram
```
Input Image/Video
       ↓
   YOLO Detection
       ↓
   Face Detection
       ↓
   Choose Mode:
   ├── 1. Blur
   ├── 2. Pixelate  
   ├── 3. Emoji
   └── 4. Synthetic Face Swap
              ├── Random faces
              └── Same face
       ↓
   Output Image/Video
```

### Technical Details

1. **YOLO Detection**: Uses YOLOv12l-face for accurate face detection
2. **Face Processing**: Applies chosen anonymization to detected faces
3. **Synthetic Replacement**: 
   - **High-quality**: Uses InSwapper for realistic face replacement
   - **Fallback**: Uses classic OpenCV method if InSwapper unavailable
4. **Options**: Random faces, same face, keep main subject

## 🎨 Mode Details

### 1. Blur Mode
- **Method**: Multiple blur types with adaptive kernel
  - **Gaussian**: Smooth blur with configurable radius (σ)
  - **Box**: Uniform blur for consistent effect
  - **Median**: Edge-preserving blur for noise reduction
- **Speed**: Fast
- **Quality**: Good for basic anonymization
- **Parameters**: 
  - `--blur_type`: gaussian, box, median
  - `--blur_strength`: Kernel size (0=auto)
  - `--blur_radius`: Gaussian blur radius (σ)
- **Use case**: Quick privacy protection with customizable effects

### 2. Pixelate Mode
- **Method**: Advanced pixelation with multiple interpolation methods
  - **Nearest**: Classic blocky pixelation
  - **Linear**: Smoother pixelation with linear interpolation
  - **Cubic**: High-quality pixelation with cubic interpolation
- **Speed**: Fast
- **Quality**: Medium to High (configurable quality)
- **Parameters**:
  - `--pixelate_method`: nearest, linear, cubic
  - `--blocks`: Block size (4-20)
  - `--pixelate_quality`: Quality percentage (10-100%)
- **Use case**: Classic pixelation effect with enhanced control

### 3. Emoji Mode
- **Method**: Overlay PNG emoji with transparency
- **Speed**: Fast
- **Quality**: Fun and creative
- **Available Emojis**: 5 built-in options (Baozou, Cat, Doge, Shrek, Yao)
- **GUI Features**: Radio button selection with live preview
- **Use case**: Entertainment, social media, creative content

### 4. Synthetic Face Swap (High-Quality InSwapper)
- **Method**: High-quality face replacement using InsightFace InSwapper
- **Speed**: Slower (high quality)
- **Quality**: Excellent (realistic)
- **Options**: 
  - `--random_faces`: Different synthetic face for each person
  - `--keep_largest`: Preserve main subject
- **Use case**: Professional anonymization, privacy protection
- **Fallback**: Automatically falls back to classic method if InSwapper unavailable

## 💡 Tips

1. **For best results**: Use neutral, front-facing synthetic faces (256px+)
2. **Performance**: GPU acceleration recommended for video processing
3. **Quality vs Speed**: 
   - Blur/Pixelate: Fast, basic quality
   - Synthetic: Slower, high quality
4. **Synthetic faces**: Use `--random_faces` for variety, omit for consistency
5. **Main subject**: Use `--keep_largest` to preserve the primary person
6. **Emoji selection**: In GUI, use radio buttons to choose from 5 built-in emojis with live preview
7. **Blur types**: Try different blur types for different effects:
   - Gaussian: Smooth, natural blur
   - Box: Uniform, consistent blur
   - Median: Edge-preserving, noise-reducing blur
8. **Pixelation methods**: Choose interpolation for different pixelation styles:
   - Nearest: Classic blocky effect
   - Linear: Smoother transitions
   - Cubic: High-quality, detailed pixelation

## 🛠️ Troubleshooting

### Model Download Issues
```bash
# Re-download YOLO model
curl -L -o yolov12l-face.pt "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12l-face.pt"

# Re-download InSwapper model
gdown "https://drive.google.com/uc?id=1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF" \
  -O ~/.insightface/models/inswapper_128.onnx
```

### Dependencies Issues
```bash
# Update pip and reinstall
pip install --upgrade pip
pip install -r requirements.txt
```

### Performance Issues
- Use `--threshold 0.7` for faster detection (fewer false positives)
- Use `--blur_strength 15` for stronger blur
- Use `--blocks 8` for more pixelation

## 📄 License

This project is part of the TikTok Tech Jam challenge.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!
