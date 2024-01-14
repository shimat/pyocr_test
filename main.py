import os
import cv2
import numpy as np
import pyocr
from pathlib import Path
from PIL import Image, ImageEnhance

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR"
TESSDATA_PATH = r"C:\Program Files\Tesseract-OCR\tessdata"

os.environ["PATH"] += os.pathsep + TESSERACT_PATH
os.environ["TESSDATA_PREFIX"] = TESSDATA_PATH


def enhance_contrast(img: Image, factor: float = 2.0) -> Image:
    enhancer = ImageEnhance.Contrast(img)
    img_enhanced = enhancer.enhance(factor)
    return img_enhanced


def get_contours(img: Image):
    cvimg_gray = np.array(img, dtype=np.uint8)
    assert(cvimg_gray.ndim == 2)

    _, cvimg_binary = cv2.threshold(cvimg_gray, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite("output/opencv_binary.png", cvimg_binary)

    contours, hierarchy = cv2.findContours(
        cvimg_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cvimg_blank = np.ones_like(cvimg_gray) * 255
    cv2.drawContours(cvimg_blank, contours, -1, (0,0,0), 3)
    cv2.imwrite("output/opencv_contours.png", cvimg_blank)


def main():
    INPUT_DIR = Path("input")
    TARGET_IMAGE_PATH = "input/test1.jpg"

    tool: pyocr.tesseract = pyocr.get_available_tools()[0]

    # 7. Treat the image as a single text line.
    builder = pyocr.builders.TextBuilder(tesseract_layout=7)

    os.makedirs("output", exist_ok=True)

    for file in INPUT_DIR.glob("*.jpg"):
        print(file)
        img = Image.open(file)
        img_gray = img.convert('L')
        img_address = img_gray.crop((0, 170, 1080, 310))
        img_kokudaka = img_gray.crop((0, 1400, 1080, 1580))

        img_address_contrast = enhance_contrast(img_address, 2.0)
        img_kokudaka_contrast = enhance_contrast(img_kokudaka, 2.0)

        img_address.save("output/address.png")
        img_kokudaka.save("output/kokudaka.png")
        img_address_contrast.save("output/address_contrast.png")
        img_kokudaka_contrast.save("output/kokudaka_contrast.png")

        get_contours(img_address_contrast)
        get_contours(img_kokudaka_contrast)

        text_address = tool.image_to_string(img_address_contrast, lang='jpn', builder=builder)
        text_kokudaka = tool.image_to_string(img_kokudaka_contrast, lang='jpn', builder=builder)
        print(f"{text_address=}")
        print(f"{text_kokudaka=}")
        #break


if __name__ == "__main__":
    main()
