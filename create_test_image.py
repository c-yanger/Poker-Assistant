#!/usr/bin/env python3
"""
Create a test image with card text for testing OCR functionality
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_test_card_image(output_path: str = "test_cards.jpg"):
    """Create a test image with card-like text"""
    
    # Create a white background
    width, height = 800, 600
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Convert to PIL for text rendering
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Try to use a default font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 60)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
        except:
            font = ImageFont.load_default()
    
    # Card positions and labels
    cards = [
        ("As", 100, 100),   # Ace of Spades
        ("Kh", 300, 100),   # King of Hearts
        ("Qd", 500, 100),   # Queen of Diamonds
        ("Jc", 700, 100),   # Jack of Clubs
        ("10s", 100, 300),  # 10 of Spades
        ("9h", 300, 300),   # 9 of Hearts
        ("8d", 500, 300),   # 8 of Diamonds
        ("7c", 700, 300),   # 7 of Clubs
    ]
    
    # Draw card rectangles and text
    for card_label, x, y in cards:
        # Draw card rectangle
        card_width, card_height = 80, 120
        
        # Draw white card background
        draw.rectangle([x-5, y-5, x+card_width+5, y+card_height+5], 
                      fill='white', outline='black', width=2)
        
        # Draw card text
        draw.text((x+10, y+10), card_label, fill='black', font=font)
        
        # Draw suit symbols for better recognition
        suit_map = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
        suit_char = card_label[-1].lower()
        if suit_char in suit_map:
            draw.text((x+10, y+60), suit_map[suit_char], fill='black', font=font)
    
    # Convert back to numpy array
    final_image = np.array(pil_image)
    
    # Save the image
    cv2.imwrite(output_path, final_image)
    print(f"Test image created: {output_path}")
    
    return output_path

def create_realistic_test_image(output_path: str = "test_realistic_cards.jpg"):
    """Create a more realistic test image with noise and variations"""
    
    # Create the basic test image first
    test_image = create_test_card_image("temp_test.jpg")
    
    # Load the image
    image = cv2.imread(test_image)
    
    # Add some noise and variations to make it more realistic
    
    # 1. Add Gaussian noise
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    # 2. Apply slight blur
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # 3. Adjust brightness and contrast slightly
    alpha = 1.1  # Contrast control
    beta = 10    # Brightness control
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # 4. Add some perspective distortion
    rows, cols, _ = image.shape
    
    # Define source and destination points for perspective transform
    src_points = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    dst_points = np.float32([[10, 5], [cols-5, 10], [5, rows-10], [cols-10, rows-5]])
    
    # Apply perspective transform
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    image = cv2.warpPerspective(image, matrix, (cols, rows))
    
    # Save the realistic test image
    cv2.imwrite(output_path, image)
    print(f"Realistic test image created: {output_path}")
    
    # Clean up temporary file
    if os.path.exists("temp_test.jpg"):
        os.remove("temp_test.jpg")
    
    return output_path

def main():
    """Create test images for OCR testing"""
    print("Creating test images for OCR testing...")
    
    # Create a simple test image
    simple_image = create_test_card_image("test_cards_simple.jpg")
    
    # Create a realistic test image with noise
    realistic_image = create_realistic_test_image("test_cards_realistic.jpg")
    
    print("\nTest images created successfully!")
    print(f"Simple test image: {simple_image}")
    print(f"Realistic test image: {realistic_image}")
    
    print("\nYou can now test the improved OCR system with:")
    print("  python poker/main.py test_cards_simple.jpg")
    print("  python poker/main.py test_cards_realistic.jpg")

if __name__ == "__main__":
    main()