import sys
import os
from ocr_detection import ImprovedCardDetector

def simple_hand_evaluation(cards):
    """Simple placeholder for hand evaluation - you can integrate with deuces later"""
    # This is a simplified version - integrate with deuces library for full evaluation
    if len(cards) < 5:
        return "Incomplete Hand", cards
    
    # For now, just return the cards
    return "Hand Detected", cards[:5]

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_image>")
        print("Options:")
        print("  --ocr-only       Use only OCR detection")
        print("  --template-only  Use only template matching")
        print("  --visualize      Save visualization of detected cards")
        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    # Parse options
    use_ocr = "--template-only" not in sys.argv
    use_templates = "--ocr-only" not in sys.argv
    visualize = "--visualize" in sys.argv

    # Initialize detector
    template_folder = os.path.join(os.path.dirname(__file__), "..", "card_templates")
    detector = ImprovedCardDetector(template_folder)
    
    # Detect cards
    print(f"Processing image: {image_path}")
    print(f"Detection mode: {'OCR' if use_ocr and not use_templates else 'Template' if use_templates and not use_ocr else 'Hybrid (OCR + Template)'}")
    
    detections = detector.detect_cards(image_path, use_ocr=use_ocr, use_templates=use_templates)
    
    if not detections:
        print("No cards detected in the image.")
        return
    
    # Display detected cards
    print(f"\nDetected {len(detections)} cards:")
    cards = []
    for det in detections:
        card = det['card']
        method = det['method']
        confidence = det['confidence']
        print(f"  - {card} (Method: {method}, Confidence: {confidence:.2%})")
        cards.append(card)
    
    # Evaluate poker hand
    if len(cards) >= 5:
        hand_type, best_cards = simple_hand_evaluation(cards)
        print(f"\nBest Hand: {hand_type}")
        print(f"Cards: {', '.join(best_cards)}")
    else:
        print(f"\nNeed at least 5 cards for poker hand evaluation. Only found {len(cards)}.")
    
    # Save visualization if requested
    if visualize:
        output_path = f"detected_{os.path.basename(image_path)}"
        detector.visualize_detections(image_path, detections, output_path)
        print(f"\nVisualization saved to: {output_path}")

if __name__ == "__main__":
    main()
