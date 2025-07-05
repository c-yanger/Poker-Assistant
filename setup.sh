#!/bin/bash

echo "Setting up Poker Card OCR Detection System"
echo "=========================================="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux system"
    
    # Update package list
    echo "Updating package list..."
    sudo apt-get update
    
    # Install Tesseract OCR
    echo "Installing Tesseract OCR..."
    sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
    
    # Install Python dependencies
    echo "Installing Python packages..."
    pip install -r requirements.txt
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS system"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    # Install Tesseract OCR
    echo "Installing Tesseract OCR..."
    brew install tesseract
    
    # Install Python dependencies
    echo "Installing Python packages..."
    pip install -r requirements.txt
    
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "Detected Windows system"
    echo "Please install Tesseract manually from:"
    echo "  https://github.com/UB-Mannheim/tesseract/wiki"
    echo "Then run: pip install -r requirements.txt"
    exit 1
else
    echo "Unknown OS type: $OSTYPE"
    exit 1
fi

# Verify Tesseract installation
echo ""
echo "Verifying Tesseract installation..."
if command -v tesseract &> /dev/null; then
    tesseract --version
    echo "✓ Tesseract installed successfully"
else
    echo "✗ Tesseract installation failed"
    exit 1
fi

# Test Python imports
echo ""
echo "Testing Python imports..."
python -c "import cv2; print('✓ OpenCV imported successfully')" 2>/dev/null || echo "✗ OpenCV import failed"
python -c "import numpy; print('✓ NumPy imported successfully')" 2>/dev/null || echo "✗ NumPy import failed"
python -c "import pytesseract; print('✓ PyTesseract imported successfully')" 2>/dev/null || echo "✗ PyTesseract import failed"
python -c "from PIL import Image; print('✓ PIL imported successfully')" 2>/dev/null || echo "✗ PIL import failed"

echo ""
echo "Setup complete! You can now run:"
echo "  python poker/main.py <image_path> --visualize"
echo "  python test_ocr_improvements.py"