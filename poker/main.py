import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_image>")
        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    # This is where your actual detection logic would be called
    print(f"Processing image: {image_path}")
    print("Detected Cards: Ah, Kd, Qs, Jc, 10h")
    print("Best Hand: Royal Flush")
    print("Confidence: High")

if __name__ == "__main__":
    main()
