import os
import cv2
import pyocr
from pathlib import Path
from PIL import Image, ImageEnhance

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR"
TESSDATA_PATH = r"C:\Program Files\Tesseract-OCR\tessdata"

os.environ["PATH"] += os.pathsep + TESSERACT_PATH
os.environ["TESSDATA_PREFIX"] = TESSDATA_PATH

TARGET_IMAGE_PATH = "input/test1.jpg"


tool: pyocr.tesseract = pyocr.get_available_tools()[0]

builder = pyocr.builders.TextBuilder(tesseract_layout=6)

img = Image.open(TARGET_IMAGE_PATH)
img_g = img.convert('L')
enhancer = ImageEnhance.Contrast(img_g)
img_con = enhancer.enhance(2.0)

#txt_pyocr = tools[0].image_to_string(img_con, lang='jpn', builder=builder)
#print(txt_pyocr)

txt = tool.image_to_string(
    img_g,
    lang='jpn+eng',
    builder=pyocr.builders.TextBuilder(tesseract_layout=6)
)
print(txt)
Path("result.txt").write_text(txt, encoding="utf-8-sig")

word_boxes = tool.image_to_string(
    img_g,
    lang='jpn+eng',
    builder=pyocr.builders.WordBoxBuilder(tesseract_layout=6)
)

cvimg = cv2.imread(TARGET_IMAGE_PATH)
for box in word_boxes:
    cv2.rectangle(cvimg, box.position[0], box.position[1], (255, 0, 0), 1)
cv2.imwrite('view.png', cvimg)
