import os
import cv2
import numpy as np
import pyocr
from pathlib import Path
from PIL import Image, ImageEnhance, ImageDraw, ImageFont

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR"
TESSDATA_PATH = r"C:\Program Files\Tesseract-OCR\tessdata"

os.environ["PATH"] += os.pathsep + TESSERACT_PATH
os.environ["TESSDATA_PREFIX"] = TESSDATA_PATH


def enhance_contrast(img: Image, factor: float = 2.0) -> Image:
    enhancer = ImageEnhance.Contrast(img)
    img_enhanced = enhancer.enhance(factor)
    return img_enhanced


def get_contours(img: Image, file_index: int):
    cvimg_gray = np.array(img, dtype=np.uint8)
    assert(cvimg_gray.ndim == 2)

    _, cvimg_binary = cv2.threshold(cvimg_gray, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"output/{file_index}_opencv_binary.png", cvimg_binary)

    contours, hierarchy = cv2.findContours(
        cvimg_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cvimg_blank = np.ones_like(cvimg_gray) * 255
    cv2.drawContours(cvimg_blank, contours, -1, (0,0,0), 3)
    cv2.imwrite(f"output/{file_index}_opencv_contours.png", cvimg_blank)


def get_boxes(
    img: Image,
    img_org: Image,
    tool: pyocr.tesseract,
    file_index: int,
    type_: str):
    word_boxes = tool.image_to_string(
        img,
        lang="jpn",
        builder=pyocr.builders.WordBoxBuilder(tesseract_layout=7)
    )
    img_boxes = img_org.copy()
    draw = ImageDraw.Draw(img_boxes)
    font = ImageFont.truetype(r"C:\Windows\Fonts\meiryo.ttc", 24)
    for box in word_boxes:
        # print(box)
        x, y = box.position[0]
        text = box.content
        color = (255, 0, 0)
        draw.text((x, y-20), text, fill=color, font=font)
    img_boxes.save(f"output/{file_index}_boxes_{type_}.png")


def main():
    INPUT_DIR = Path("input")
    TARGET_IMAGE_PATH = "input/test1.jpg"

    tool: pyocr.tesseract = pyocr.get_available_tools()[0]
    builder = pyocr.builders.TextBuilder(tesseract_layout=7)  # 7. Treat the image as a single text line.

    os.makedirs("output", exist_ok=True)

    for file_index, file in enumerate(INPUT_DIR.glob("*.jpg")):
        print(file)
        img = Image.open(file)
        img_address = img.crop((0, 170, 1080, 310))
        img_kokudaka = img.crop((0, 1400, 1080, 1580))
        img_address_gray = img_address.convert('L')
        img_kokudaka_gray = img_kokudaka.convert('L')

        img_address_contrast = enhance_contrast(img_address_gray, 2.0)
        img_kokudaka_contrast = enhance_contrast(img_kokudaka_gray, 2.0)

        img_address.save(f"output/{file_index}_address.png")
        img_kokudaka.save(f"output/{file_index}_kokudaka.png")
        img_address_contrast.save(f"output/{file_index}_address_contrast.png")
        img_kokudaka_contrast.save(f"output/{file_index}_kokudaka_contrast.png")

        # get_contours(img_address_contrast, file_index)
        # get_contours(img_kokudaka_contrast, file_index)

        text_address = tool.image_to_string(img_address_contrast, lang='jpn', builder=builder)
        text_kokudaka = tool.image_to_string(img_kokudaka_contrast, lang='jpn', builder=builder)
        print(f"{text_address=}")
        print(f"{text_kokudaka=}")

        get_boxes(img_address_contrast, img_address, tool, file_index, "address")
        get_boxes(img_kokudaka_contrast, img_kokudaka, tool, file_index, "kokudaka")




if __name__ == "__main__":
    main()
