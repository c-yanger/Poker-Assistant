import sys
import os
from ocr_only_detector import OCROnlyCardDetector
from evaluator import parse_card_input
from deuces import Card, Evaluator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_image>")
        print("Example: python main.py ../test_cards.jpg")
        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    # Initialize the OCR-only card detector
    detector = OCROnlyCardDetector()
    
    print(f"Processing image: {image_path}")
    print("=" * 50)
    
    # Detect cards using OCR
    results = detector.detect_cards(image_path)
    
    if not results:
        print("No cards detected in the image.")
        print("Tips for better detection:")
        print("- Ensure good lighting and contrast")
        print("- Make sure card text is clearly visible")
        print("- Try different image angles or distances")
        return
    
    print(f"Detected {len(results)} cards:")
    detected_cards = []
    
    for card, confidence in results:
        print(f"  {card}: {confidence:.3f} confidence")
        detected_cards.append(card)
    
    # If we have at least 5 cards, try to evaluate the best hand
    if len(detected_cards) >= 5:
        print("\n" + "=" * 50)
        print("HAND EVALUATION")
        print("=" * 50)
        
        try:
            # Convert detected cards to Card objects
            card_objects = []
            for card_str in detected_cards:
                try:
                    card_obj = Card.new(card_str)
                    card_objects.append(card_obj)
                except Exception as e:
                    logger.warning(f"Could not parse card {card_str}: {e}")
            
            if len(card_objects) >= 5:
                # Initialize evaluator
                evaluator = Evaluator()
                
                # For poker evaluation, we need exactly 5 cards
                # If we have more than 5, we'll evaluate the best 5-card hand
                if len(card_objects) == 5:
                    hand_cards = card_objects[:2]  # Player's hand
                    board_cards = card_objects[2:] # Community cards
                elif len(card_objects) == 7:
                    hand_cards = card_objects[:2]  # Player's hand
                    board_cards = card_objects[2:] # Community cards
                else:
                    # If we don't have standard poker format, just use first 5 cards
                    hand_cards = card_objects[:2]
                    board_cards = card_objects[2:5]
                
                # Evaluate the hand
                score = evaluator.evaluate(hand_cards, board_cards)
                rank_class = evaluator.get_rank_class(score)
                hand_type = evaluator.class_to_string(rank_class)
                
                print(f"Hand Type: {hand_type}")
                print(f"Hand Strength: {score} (lower is better)")
                
                # Calculate hand strength percentage
                hand_strength = 1.0 - (score / 7462.0)  # 7462 is worst possible score
                confidence_level = "High" if hand_strength > 0.8 else "Medium" if hand_strength > 0.5 else "Low"
                
                print(f"Hand Strength: {hand_strength:.1%}")
                print(f"Confidence: {confidence_level}")
                
                print("\nYour cards:")
                Card.print_pretty_cards(hand_cards)
                
                print("Community cards:")
                Card.print_pretty_cards(board_cards)
                
            else:
                print("Need at least 5 valid cards for hand evaluation")
                
        except Exception as e:
            logger.error(f"Error evaluating hand: {e}")
            print("Could not evaluate hand - invalid card format")
    
    else:
        print(f"\nNeed at least 5 cards for hand evaluation (found {len(detected_cards)})")
    
    print("\n" + "=" * 50)
    print("DETECTION SUMMARY")
    print("=" * 50)
    
    # Summarize detection quality
    avg_confidence = sum(conf for _, conf in results) / len(results) if results else 0
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Total cards detected: {len(results)}")
    
    if avg_confidence > 0.8:
        print("Detection quality: Excellent")
    elif avg_confidence > 0.6:
        print("Detection quality: Good")
    elif avg_confidence > 0.4:
        print("Detection quality: Fair")
    else:
        print("Detection quality: Poor - consider improving image quality")

if __name__ == "__main__":
    main()
