#!/usr/bin/env python3
"""
Test script for improved OCR card detection
This script demonstrates various techniques to improve OCR accuracy
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

def test_ocr_configurations():
    """Test different OCR configurations and preprocessing methods"""
    
    print("OCR Improvement Test Suite")
    print("=" * 50)
    
    # Test image
    test_image_path = "test_image.png"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found at: {test_image_path}")
        print("Please provide a test image with playing cards")
        return
    
    # Load image
    img = cv2.imread(test_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"\nTesting OCR on: {test_image_path}")
    print(f"Image dimensions: {img.shape}")
    
    # Test 1: Basic OCR
    print("\n1. Basic OCR (no preprocessing):")
    basic_text = pytesseract.image_to_string(gray)
    print(f"   Result: {basic_text.strip() if basic_text.strip() else 'No text detected'}")
    
    # Test 2: With thresholding
    print("\n2. OCR with Binary Thresholding:")
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    thresh_text = pytesseract.image_to_string(thresh)
    print(f"   Result: {thresh_text.strip() if thresh_text.strip() else 'No text detected'}")
    
    # Test 3: Adaptive thresholding
    print("\n3. OCR with Adaptive Thresholding:")
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
    adaptive_text = pytesseract.image_to_string(adaptive_thresh)
    print(f"   Result: {adaptive_text.strip() if adaptive_text.strip() else 'No text detected'}")
    
    # Test 4: With denoising
    print("\n4. OCR with Denoising:")
    denoised = cv2.fastNlMeansDenoising(gray)
    denoised_text = pytesseract.image_to_string(denoised)
    print(f"   Result: {denoised_text.strip() if denoised_text.strip() else 'No text detected'}")
    
    # Test 5: With morphological operations
    print("\n5. OCR with Morphological Operations:")
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph_text = pytesseract.image_to_string(morph)
    print(f"   Result: {morph_text.strip() if morph_text.strip() else 'No text detected'}")
    
    # Test 6: With upscaling
    print("\n6. OCR with Image Upscaling (3x):")
    upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    upscaled_text = pytesseract.image_to_string(upscaled)
    print(f"   Result: {upscaled_text.strip() if upscaled_text.strip() else 'No text detected'}")
    
    # Test 7: Card-specific configuration
    print("\n7. OCR with Card-Specific Configuration:")
    # Whitelist only card characters
    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=AKQJ234567890♠♣♥♦'
    card_text = pytesseract.image_to_string(upscaled, config=custom_config)
    print(f"   Result: {card_text.strip() if card_text.strip() else 'No text detected'}")
    
    # Test 8: Combined preprocessing
    print("\n8. OCR with Combined Preprocessing (Best Practice):")
    # Apply multiple preprocessing steps
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    processed = cv2.medianBlur(processed, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    processed = clahe.apply(processed)
    processed = cv2.resize(processed, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    combined_text = pytesseract.image_to_string(processed, config=custom_config)
    print(f"   Result: {combined_text.strip() if combined_text.strip() else 'No text detected'}")
    
    # Save preprocessed images for comparison
    print("\n9. Saving preprocessed images for visual comparison:")
    cv2.imwrite("ocr_test_original.png", gray)
    cv2.imwrite("ocr_test_threshold.png", thresh)
    cv2.imwrite("ocr_test_adaptive.png", adaptive_thresh)
    cv2.imwrite("ocr_test_denoised.png", denoised)
    cv2.imwrite("ocr_test_morph.png", morph)
    cv2.imwrite("ocr_test_upscaled.png", upscaled)
    cv2.imwrite("ocr_test_combined.png", processed)
    print("   Saved preprocessed images for comparison")
    
    # OCR Tips
    print("\n" + "=" * 50)
    print("OCR Improvement Tips:")
    print("=" * 50)
    print("1. Image Quality:")
    print("   - Use high-resolution images (300 DPI or higher)")
    print("   - Ensure good lighting and contrast")
    print("   - Avoid blurry or skewed images")
    print("\n2. Preprocessing:")
    print("   - Convert to grayscale")
    print("   - Apply adaptive thresholding for varying lighting")
    print("   - Use denoising for noisy images")
    print("   - Upscale small text (3x-5x)")
    print("\n3. OCR Configuration:")
    print("   - Use character whitelist for known characters")
    print("   - Choose appropriate PSM (Page Segmentation Mode)")
    print("   - PSM 8: Single word")
    print("   - PSM 11: Sparse text")
    print("   - PSM 13: Raw line")
    print("\n4. Card-Specific:")
    print("   - Focus on card corners where rank/suit are displayed")
    print("   - Use template matching as fallback")
    print("   - Train custom OCR model for playing cards if needed")
    print("\n5. Hybrid Approach:")
    print("   - Combine OCR with template matching")
    print("   - Use confidence scores to choose best method")
    print("   - Implement validation logic for card values")


def benchmark_ocr_speed():
    """Benchmark OCR performance with different settings"""
    import time
    
    print("\n\nOCR Speed Benchmark")
    print("=" * 50)
    
    test_image_path = "test_image.png"
    if not os.path.exists(test_image_path):
        print("Test image not found")
        return
    
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Test different image sizes
    sizes = [(1, "Original"), (2, "2x Upscaled"), (3, "3x Upscaled"), (4, "4x Upscaled")]
    
    for scale, name in sizes:
        if scale > 1:
            test_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            test_img = img
            
        start_time = time.time()
        text = pytesseract.image_to_string(test_img)
        end_time = time.time()
        
        print(f"{name}: {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    # Check if Tesseract is installed
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {tesseract_version}")
    except:
        print("ERROR: Tesseract not found. Please install Tesseract OCR:")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("  macOS: brew install tesseract")
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        exit(1)
    
    # Run tests
    test_ocr_configurations()
    benchmark_ocr_speed()