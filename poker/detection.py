import cv2
import os

def load_templates(template_folder):
    templates = {}
    for filename in os.listdir(template_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            label = os.path.splitext(filename)[0]
            path = os.path.join(template_folder, filename)
            templates[label] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return templates

def match_cards(image_path, templates, threshold=0.8):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # ✅ Step 1: Check if the image loaded correctly
    if img is None:
        print(f"[ERROR] Could not load image at: {image_path}")
        return []

    matches = []
    # (existing matching logic continues here)


    for label, template in templates.items():
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        loc = cv2.minMaxLoc(res)
        max_val = loc[1]
        if max_val > threshold:
            matches.append((label, max_val))

    return sorted(matches, key=lambda x: -x[1])  # Best match first

if __name__ == "__main__":
    templates = load_templates("card_templates")
    detected = match_cards("test_hand.jpg", templates)

    print("\n[DEBUG] Raw match scores:")
    for label, score in detected:
        print(f"Checked: {label} -> {score:.2f}")

    print("\nDetected cards:")
    if not detected:
        print("[INFO] No cards detected above the threshold.")
    else:
        for label, score in detected:
            if score >= 0.8:
                print(f"{label} ({score:.2f})")
            else:
                print("[INFO] Matches found, but below confidence threshold.")

