# Poker Assistant 🃏💻

A computer vision-based poker assistant that detects cards from an input image, extracts their values using OCR, and determines the strength of the poker hand. Built for practical use or experimentation with AI-powered gameplay assistants.

## 🚀 Features

- 🖼️ Detects and identifies poker cards in real images
- 🔍 Uses Tesseract OCR to read card values
- 🧠 Evaluates hand strength (e.g., Pair, Straight, Flush)
- 🧪 Test images provided for validation and demonstration

## 🛠️ How to Run

1. **Clone the repository**

```bash
git clone https://github.com/c-yanger/Poker-Assistant.git
cd Poker-Assistant

2. Set up virtual environment

bash
python -m venv env
.\env\Scripts\activate

3. Install dependencies

bash
pip install -r requirements.txt

4. Run the Assistant 

bash
python poker/main.py -i examples/sample_hand_1.png


📦 Tech Stack

Python 3

OpenCV

Pytesseract (Tesseract OCR)

NumPy

Custom trained templates

🧭 Future Improvements

Real-time camera integration

GUI overlay for AR glasses

Better OCR accuracy via fine-tuning