import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import logging
from typing import List, Tuple, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedCardDetector:
    def __init__(self, template_folder: str, threshold: float = 0.7):
        """
        Initialize the improved card detector
        
        Args:
            template_folder: Path to card template images
            threshold: Minimum confidence threshold for template matching
        """
        self.template_folder = template_folder
        self.threshold = threshold
        self.templates = self._load_templates()
        
        # OCR configuration for better accuracy
        self.ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789AJQK'
        
        # Card suits and ranks for validation
        self.valid_ranks = {'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'}
        self.valid_suits = {'♠', '♥', '♦', '♣', 's', 'h', 'd', 'c', 'S', 'H', 'D', 'C'}
        
    def _load_templates(self) -> Dict[str, np.ndarray]:
        """Load and preprocess card templates"""
        templates = {}
        
        if not os.path.exists(self.template_folder):
            logger.error(f"Template folder not found: {self.template_folder}")
            return templates
            
        for filename in os.listdir(self.template_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                label = os.path.splitext(filename)[0]
                path = os.path.join(self.template_folder, filename)
                
                try:
                    template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        # Preprocess template
                        template = self._preprocess_template(template)
                        templates[label] = template
                        logger.info(f"Loaded template: {label}")
                    else:
                        logger.warning(f"Could not load template: {path}")
                except Exception as e:
                    logger.error(f"Error loading template {path}: {e}")
                    
        logger.info(f"Loaded {len(templates)} templates")
        return templates
        
    def _preprocess_template(self, template: np.ndarray) -> np.ndarray:
        """Preprocess template for better matching"""
        # Apply Gaussian blur to reduce noise
        template = cv2.GaussianBlur(template, (3, 3), 0)
        
        # Apply adaptive thresholding
        template = cv2.adaptiveThreshold(
            template, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return template
        
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image for better detection"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Apply Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        return image
        
    def _extract_card_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Extract potential card regions from image"""
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        card_regions = []
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Filter by area (adjust these values based on your card size)
            if 1000 < area < 50000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (playing cards are roughly 2.5:3.5)
                aspect_ratio = w / h
                if 0.6 < aspect_ratio < 0.8:
                    card_regions.append((x, y, w, h))
                    
        return card_regions
        
    def _multi_scale_template_matching(self, image: np.ndarray, template: np.ndarray) -> Tuple[float, Tuple[int, int]]:
        """Perform template matching at multiple scales"""
        best_score = 0
        best_location = (0, 0)
        
        # Try multiple scales
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        for scale in scales:
            # Resize template
            width = int(template.shape[1] * scale)
            height = int(template.shape[0] * scale)
            resized_template = cv2.resize(template, (width, height))
            
            # Skip if template is larger than image
            if resized_template.shape[0] > image.shape[0] or resized_template.shape[1] > image.shape[1]:
                continue
                
            # Perform template matching
            result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_location = max_loc
                
        return best_score, best_location
        
    def _ocr_card_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> Optional[str]:
        """Use OCR to extract text from card region"""
        x, y, w, h = region
        
        # Extract region
        card_roi = image[y:y+h, x:x+w]
        
        # Preprocess for OCR
        card_roi = cv2.resize(card_roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        card_roi = cv2.GaussianBlur(card_roi, (5, 5), 0)
        card_roi = cv2.adaptiveThreshold(card_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        try:
            # Convert to PIL Image for pytesseract
            pil_image = Image.fromarray(card_roi)
            
            # Extract text
            text = pytesseract.image_to_string(pil_image, config=self.ocr_config)
            text = text.strip().upper()
            
            # Validate and clean OCR result
            return self._validate_card_text(text)
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None
            
    def _validate_card_text(self, text: str) -> Optional[str]:
        """Validate and clean OCR text"""
        if not text:
            return None
            
        # Remove common OCR errors
        text = text.replace('0', 'O').replace('1', 'I').replace('5', 'S')
        
        # Extract potential card information
        words = text.split()
        for word in words:
            # Check for rank-suit patterns
            if len(word) >= 2:
                rank = word[:-1]
                suit = word[-1]
                
                # Map suit characters
                suit_map = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c'}
                if suit in suit_map:
                    suit = suit_map[suit]
                    
                if rank in self.valid_ranks and suit in self.valid_suits:
                    return f"{rank}{suit}"
                    
        return None
        
    def detect_cards(self, image_path: str) -> List[Tuple[str, float, str]]:
        """
        Detect cards in image using improved methods
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of tuples: (card_label, confidence, detection_method)
        """
        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return []
            
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        detections = []
        
        # Method 1: Improved template matching
        template_results = self._template_matching_detection(processed_image)
        detections.extend([(card, score, "template") for card, score in template_results])
        
        # Method 2: OCR-based detection
        ocr_results = self._ocr_based_detection(processed_image)
        detections.extend([(card, 0.8, "ocr") for card in ocr_results if card])
        
        # Remove duplicates and sort by confidence
        unique_detections = {}
        for card, score, method in detections:
            if card not in unique_detections or score > unique_detections[card][1]:
                unique_detections[card] = (card, score, method)
                
        final_results = list(unique_detections.values())
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Detected {len(final_results)} cards")
        return final_results
        
    def _template_matching_detection(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """Template matching with improvements"""
        matches = []
        
        for label, template in self.templates.items():
            score, location = self._multi_scale_template_matching(image, template)
            
            if score > self.threshold:
                matches.append((label, score))
                logger.info(f"Template match: {label} ({score:.3f})")
                
        return matches
        
    def _ocr_based_detection(self, image: np.ndarray) -> List[str]:
        """OCR-based card detection"""
        # Extract potential card regions
        card_regions = self._extract_card_regions(image)
        
        ocr_results = []
        for region in card_regions:
            card_text = self._ocr_card_region(image, region)
            if card_text:
                ocr_results.append(card_text)
                logger.info(f"OCR detection: {card_text}")
                
        return ocr_results

def main():
    """Test the improved card detector"""
    detector = ImprovedCardDetector(
        template_folder="../card_templates", 
        threshold=0.7
    )
    
    # Test with a sample image (if available)
    test_images = ["test_hand.jpg", "test_image.png", "sample_cards.jpg"]
    
    for test_image in test_images:
        if os.path.exists(test_image):
            logger.info(f"Testing with image: {test_image}")
            results = detector.detect_cards(test_image)
            
            print(f"\nResults for {test_image}:")
            if results:
                for card, confidence, method in results:
                    print(f"  {card}: {confidence:.3f} ({method})")
            else:
                print("  No cards detected")
            break
    else:
        logger.warning("No test images found")

if __name__ == "__main__":
    main()