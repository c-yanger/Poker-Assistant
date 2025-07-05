import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import logging
from typing import List, Tuple, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCROnlyCardDetector:
    def __init__(self):
        """
        Initialize the OCR-only card detector
        """
        # OCR configuration for better accuracy
        self.ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789AJQK♠♥♦♣SsHhDdCc'
        
        # Card suits and ranks for validation
        self.valid_ranks = {'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'}
        self.valid_suits = {'♠', '♥', '♦', '♣', 's', 'h', 'd', 'c', 'S', 'H', 'D', 'C'}
        
        # OCR error correction mappings
        self.rank_corrections = {
            '0': 'O',
            '1': 'I',
            '5': 'S',
            'G': '6',
            'B': '8',
            'T': '7',
            'l': '1',
            'O': '0',
            'I': '1'
        }
        
        # Suit symbol mappings
        self.suit_symbols = {
            '♠': 's', '♥': 'h', '♦': 'd', '♣': 'c',
            'S': 's', 'H': 'h', 'D': 'd', 'C': 'c',
            'SPADES': 's', 'HEARTS': 'h', 'DIAMONDS': 'd', 'CLUBS': 'c'
        }
        
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image for better OCR"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction
        image = cv2.medianBlur(image, 5)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        return image
    
    def _extract_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Extract text regions from the image"""
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Filter by area (text regions should be medium-sized)
            if 500 < area < 20000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 3.0:  # More flexible for text
                    text_regions.append((x, y, w, h))
        
        return text_regions
    
    def _extract_card_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Extract potential card regions from image"""
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        card_regions = []
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Filter by area (cards should be reasonably large)
            if 2000 < area < 100000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (playing cards are roughly rectangular)
                aspect_ratio = w / h
                if 0.4 < aspect_ratio < 1.5:  # More flexible range
                    card_regions.append((x, y, w, h))
        
        return card_regions
    
    def _ocr_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> str:
        """Extract text from a specific region using OCR"""
        x, y, w, h = region
        
        # Extract region with some padding
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        roi = image[y1:y2, x1:x2]
        
        # Enhance the ROI for better OCR
        roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # Apply additional preprocessing
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
        
        try:
            # Convert to PIL Image for pytesseract
            pil_image = Image.fromarray(roi)
            
            # Extract text using OCR
            text = pytesseract.image_to_string(pil_image, config=self.ocr_config)
            text = text.strip()
            
            return text
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def _clean_and_validate_text(self, text: str) -> List[str]:
        """Clean OCR text and extract valid card representations"""
        if not text:
            return []
        
        # Remove unwanted characters and normalize
        text = re.sub(r'[^\w♠♥♦♣]', ' ', text)
        text = text.upper()
        
        # Split into words
        words = text.split()
        
        valid_cards = []
        
        for word in words:
            # Try to extract card information
            card = self._extract_card_from_word(word)
            if card:
                valid_cards.append(card)
        
        # Also try to find patterns in the full text
        patterns = self._find_card_patterns(text)
        valid_cards.extend(patterns)
        
        # Remove duplicates while preserving order
        unique_cards = []
        seen = set()
        for card in valid_cards:
            if card not in seen:
                unique_cards.append(card)
                seen.add(card)
        
        return unique_cards
    
    def _extract_card_from_word(self, word: str) -> Optional[str]:
        """Extract card information from a single word"""
        if len(word) < 2:
            return None
        
        # Apply corrections
        corrected_word = word
        for wrong, right in self.rank_corrections.items():
            corrected_word = corrected_word.replace(wrong, right)
        
        # Try different patterns
        patterns = [
            r'([A-Z0-9]{1,2})([♠♥♦♣SHDC])',  # Rank + suit
            r'([A-Z0-9]{1,2})([♠♥♦♣SHDC])',  # Alternative pattern
        ]
        
        for pattern in patterns:
            match = re.search(pattern, corrected_word)
            if match:
                rank, suit = match.groups()
                
                # Validate rank
                if rank in self.valid_ranks:
                    # Convert suit to lowercase standard
                    suit_lower = self.suit_symbols.get(suit, suit.lower())
                    if suit_lower in ['s', 'h', 'd', 'c']:
                        return f"{rank}{suit_lower}"
        
        return None
    
    def _find_card_patterns(self, text: str) -> List[str]:
        """Find card patterns in the full text"""
        cards = []
        
        # Look for rank-suit combinations
        pattern = r'([A-Z0-9]{1,2})\s*([♠♥♦♣SHDC])'
        matches = re.findall(pattern, text)
        
        for rank, suit in matches:
            if rank in self.valid_ranks:
                suit_lower = self.suit_symbols.get(suit, suit.lower())
                if suit_lower in ['s', 'h', 'd', 'c']:
                    cards.append(f"{rank}{suit_lower}")
        
        return cards
    
    def detect_cards(self, image_path: str) -> List[Tuple[str, float]]:
        """
        Detect cards in image using OCR only
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of tuples: (card_label, confidence)
        """
        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return []
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        all_detected_cards = []
        
        # Method 1: OCR on the entire image
        full_text = self._ocr_region(processed_image, (0, 0, processed_image.shape[1], processed_image.shape[0]))
        full_image_cards = self._clean_and_validate_text(full_text)
        all_detected_cards.extend([(card, 0.7) for card in full_image_cards])
        
        # Method 2: OCR on detected card regions
        card_regions = self._extract_card_regions(processed_image)
        for region in card_regions:
            region_text = self._ocr_region(processed_image, region)
            region_cards = self._clean_and_validate_text(region_text)
            all_detected_cards.extend([(card, 0.8) for card in region_cards])
        
        # Method 3: OCR on detected text regions
        text_regions = self._extract_text_regions(processed_image)
        for region in text_regions:
            region_text = self._ocr_region(processed_image, region)
            region_cards = self._clean_and_validate_text(region_text)
            all_detected_cards.extend([(card, 0.6) for card in region_cards])
        
        # Remove duplicates and sort by confidence
        unique_cards = {}
        for card, confidence in all_detected_cards:
            if card not in unique_cards or confidence > unique_cards[card]:
                unique_cards[card] = confidence
        
        final_results = [(card, conf) for card, conf in unique_cards.items()]
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Detected {len(final_results)} cards using OCR")
        for card, conf in final_results:
            logger.info(f"  {card}: {conf:.3f}")
        
        return final_results

def main():
    """Test the OCR-only card detector"""
    detector = OCROnlyCardDetector()
    
    # Test with available images
    test_images = ["test_cards_simple.jpg", "test_cards_realistic.jpg", "test_hand.jpg"]
    
    for test_image in test_images:
        if os.path.exists(test_image):
            logger.info(f"Testing with image: {test_image}")
            results = detector.detect_cards(test_image)
            
            print(f"\nResults for {test_image}:")
            if results:
                for card, confidence in results:
                    print(f"  {card}: {confidence:.3f}")
            else:
                print("  No cards detected")
        else:
            logger.info(f"Image not found: {test_image}")

if __name__ == "__main__":
    main()