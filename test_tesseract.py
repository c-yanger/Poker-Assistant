import pytesseract
from PIL import Image

image_path = 'test_image.png'
img = Image.open(image_path)
text = pytesseract.image_to_string(img)
print('Text detected by Tesseract:', text)
