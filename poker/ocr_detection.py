import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import re

class ImprovedCardDetector:
    def __init__(self, template_folder):
        self.templates = self.load_templates(template_folder)
        self.card_patterns = {
            'A': 'ace', 'K': 'king', 'Q': 'queen', 'J': 'jack',
            '10': 'ten', '9': 'nine', '8': 'eight', '7': 'seven',
            '6': 'six', '5': 'five', '4': 'four', '3': 'three', '2': 'two'
        }
        self.suit_patterns = {
            '♠': 's', '♣': 'c', '♥': 'h', '♦': 'd',
            'spades': 's', 'clubs': 'c', 'hearts': 'h', 'diamonds': 'd'
        }
        
    def load_templates(self, template_folder):
        """Load card templates for initial detection"""
        templates = {}
        for filename in os.listdir(template_folder):
            if filename.endswith((".png", ".jpg")):
                label = os.path.splitext(filename)[0]
                path = os.path.join(template_folder, filename)
                template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[label] = template
        return templates
    
    def preprocess_image(self, image):
        """Enhanced preprocessing for better OCR results"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Denoise
        denoised = cv2.medianBlur(thresh, 3)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def detect_card_regions(self, image):
        """Detect individual card regions in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        card_regions = []
        for contour in contours:
            # Approximate polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Check if it's roughly rectangular (4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (cards are typically 2.5:3.5)
                aspect_ratio = w / float(h)
                if 0.5 < aspect_ratio < 0.8 and w > 50 and h > 70:
                    card_regions.append((x, y, w, h))
        
        return card_regions
    
    def extract_card_value_ocr(self, card_image):
        """Extract card value and suit using OCR"""
        # Preprocess for OCR
        processed = self.preprocess_image(card_image)
        
        # Focus on the top-left corner where rank and suit typically are
        h, w = processed.shape[:2]
        corner_region = processed[0:int(h*0.3), 0:int(w*0.3)]
        
        # Resize for better OCR
        scale_factor = 3
        corner_resized = cv2.resize(corner_region, None, fx=scale_factor, fy=scale_factor, 
                                  interpolation=cv2.INTER_CUBIC)
        
        # OCR configuration for single characters
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=AKQJ234567890♠♣♥♦'
        
        # Convert to PIL Image for pytesseract
        pil_image = Image.fromarray(corner_resized)
        
        # Perform OCR
        text = pytesseract.image_to_string(pil_image, config=custom_config).strip()
        
        # Also try with different PSM modes for better results
        if not text:
            custom_config_alt = r'--oem 3 --psm 11'
            text = pytesseract.image_to_string(pil_image, config=custom_config_alt).strip()
        
        return self.parse_card_text(text)
    
    def parse_card_text(self, text):
        """Parse OCR text to extract card rank and suit"""
        text = text.upper().strip()
        
        # Extract rank
        rank_match = re.search(r'(10|[AKQJ2-9])', text)
        rank = rank_match.group(1) if rank_match else None
        
        # Extract suit - look for suit symbols or words
        suit = None
        for symbol, code in self.suit_patterns.items():
            if symbol in text.lower():
                suit = code
                break
        
        # If no suit found, try pattern matching
        if not suit:
            if any(c in text for c in '♠♣♥♦'):
                for c in text:
                    if c in '♠♣♥♦':
                        suit = self.suit_patterns.get(c, None)
                        break
        
        return rank, suit
    
    def match_with_templates(self, card_region, threshold=0.7):
        """Fallback template matching when OCR fails"""
        best_match = None
        best_score = 0
        
        for label, template in self.templates.items():
            # Resize card region to match template size
            th, tw = template.shape[:2]
            resized = cv2.resize(card_region, (tw, th))
            
            # Template matching
            result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score and max_val > threshold:
                best_score = max_val
                best_match = label
        
        return best_match, best_score
    
    def detect_cards(self, image_path, use_ocr=True, use_templates=True):
        """Main detection method combining OCR and template matching"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not load image at: {image_path}")
            return []
        
        detected_cards = []
        
        # Detect card regions
        regions = self.detect_card_regions(img)
        
        if not regions:
            # If no regions detected, try the whole image
            regions = [(0, 0, img.shape[1], img.shape[0])]
        
        for x, y, w, h in regions:
            card_img = img[y:y+h, x:x+w]
            card_gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            
            card_info = {'region': (x, y, w, h), 'confidence': 0}
            
            # Try OCR first
            if use_ocr:
                rank, suit = self.extract_card_value_ocr(card_img)
                if rank and suit:
                    card_info['card'] = f"{rank}{suit}"
                    card_info['method'] = 'OCR'
                    card_info['confidence'] = 0.9  # High confidence for successful OCR
                    detected_cards.append(card_info)
                    continue
            
            # Fallback to template matching
            if use_templates:
                match, score = self.match_with_templates(card_gray)
                if match:
                    card_info['card'] = match
                    card_info['method'] = 'Template'
                    card_info['confidence'] = score
                    detected_cards.append(card_info)
        
        return detected_cards
    
    def visualize_detections(self, image_path, detections, output_path=None):
        """Visualize detected cards on the image"""
        img = cv2.imread(image_path)
        
        for detection in detections:
            x, y, w, h = detection['region']
            card = detection.get('card', 'Unknown')
            confidence = detection.get('confidence', 0)
            method = detection.get('method', 'Unknown')
            
            # Draw rectangle
            color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            
            # Add label
            label = f"{card} ({method}: {confidence:.2f})"
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
        
        if output_path:
            cv2.imwrite(output_path, img)
        
        return img


# Example usage and testing function
def test_improved_detection():
    """Test the improved card detection"""
    template_folder = os.path.join(os.path.dirname(__file__), "..", "card_templates")
    detector = ImprovedCardDetector(template_folder)
    
    # Test with different methods
    test_images = ["test_hand.jpg", "../test_image.png"]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting on: {img_path}")
            
            # Test OCR + Templates
            print("\n1. OCR + Template Matching:")
            detections = detector.detect_cards(img_path, use_ocr=True, use_templates=True)
            for det in detections:
                print(f"   - {det['card']} (Method: {det['method']}, Confidence: {det['confidence']:.2f})")
            
            # Visualize results
            output_path = f"detected_{os.path.basename(img_path)}"
            detector.visualize_detections(img_path, detections, output_path)
            print(f"   Visualization saved to: {output_path}")


if __name__ == "__main__":
    test_improved_detection()