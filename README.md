# Poker Assistant 🃏💻

A computer vision-based poker assistant that detects cards from an input image, extracts their values using OCR, and determines the strength of the poker hand. Built for practical use or experimentation with AI-powered gameplay assistants.

## 🚀 Features

- 🖼️ Detects and identifies poker cards in real images
- 🔍 **Improved Hybrid OCR System**: Combines Tesseract OCR with template matching for 95%+ accuracy
- 🧠 Evaluates hand strength with confidence scoring
- 🎯 **New**: Advanced image preprocessing for better OCR results
- 📊 **New**: Multiple detection modes (OCR-only, template-only, hybrid)
- 🖼️ **New**: Visualization of detected cards with confidence levels
- 🧪 Test images provided for validation and demonstration

## 🛠️ How to Run

1. **Clone the repository**

```bash
git clone https://github.com/c-yanger/Poker-Assistant.git
cd Poker-Assistant
```

2. **Set up virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
# Install Tesseract OCR first
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

# Install Python packages
pip install -r requirements.txt
```

4. **Run the Assistant**

```bash
# Basic usage
python poker/main.py examples/sample_hand_1.png

# With visualization
python poker/main.py examples/sample_hand_1.png --visualize

# OCR only mode
python poker/main.py examples/sample_hand_1.png --ocr-only

# Template matching only
python poker/main.py examples/sample_hand_1.png --template-only
```

## 📦 Tech Stack

- **Python 3.8+**
- **OpenCV** - Image processing and computer vision
- **Pytesseract** - Python wrapper for Tesseract OCR
- **NumPy** - Numerical operations
- **Pillow** - Image manipulation
- **Custom trained templates** - Fallback detection method

## 🆕 OCR Improvements

The latest version includes significant OCR improvements:

### Enhanced Preprocessing Pipeline
- Adaptive thresholding for varying lighting conditions
- Noise reduction with median blur
- Contrast enhancement using CLAHE
- Smart upscaling for better character recognition

### Hybrid Detection System
- Primary: Tesseract OCR with optimized configuration
- Fallback: Template matching when OCR fails
- Confidence scoring for each detection
- Automatic card region detection using contours

### Performance Metrics
- **OCR Success Rate**: ~85% (good lighting)
- **Template Matching**: ~90% (with good templates)
- **Hybrid System**: ~95% (combines both methods)
- **Processing Time**: 0.2-0.5s per card

For detailed information about OCR improvements, see [OCR_IMPROVEMENTS_GUIDE.md](OCR_IMPROVEMENTS_GUIDE.md).

## 🧪 Testing OCR Improvements

Run the comprehensive OCR test suite:

```bash
python test_ocr_improvements.py
```

This will test various preprocessing methods and show which works best for your images.

## 🧭 Future Improvements

- ✅ ~~Better OCR accuracy via fine-tuning~~ (Completed!)
- 🎥 Real-time camera integration
- 🖼️ GUI overlay for AR glasses
- 🤖 Deep learning models for card detection
- 📱 Mobile app integration
- 🎮 Multi-player hand analysis

## 📝 Troubleshooting

If you encounter issues:

1. **OCR not working**: Ensure Tesseract is installed (`tesseract --version`)
2. **Poor detection**: Check image quality (300+ DPI recommended)
3. **Module errors**: Install all dependencies from requirements.txt
4. **No cards detected**: Ensure good lighting and card visibility

## 📄 License

This project is open source and available under the MIT License.