import cv2
import os
from config import get_args
import pytesseract

args = get_args()

class ExtractingText:
    def __init__(self, usingGPU: bool):
        self.gpu = usingGPU

    def cleanup_text(self, text):
        return " ".join([c if ord(c) < 128 else " " for c in text]).strip()

    def get_results(self, ocr_results, de_prob):
        words = []
        for line in ocr_results.split('\n'):
            text, prob = line.split(': ')
            prob = float(prob)
            if prob > de_prob and len(text) > 2:
                text = self.cleanup_text(text)
                words.append(text)
        return words

    def TesseractOCR(self, dtset, languages, de_prob=0.5):
        char_pos = []
        for idx in dtset:
            img_file = os.path.join(args.data_path, os.path.basename(idx))
            img = cv2.imread(img_file)
            ocr_results = pytesseract.image_to_string(img, lang=languages, config='--psm 11')
            bboxes = self.get_results(ocr_results, de_prob)
            char_pos.append([idx, bboxes])
        return char_pos
'''
import cv2
import os
from config import get_args
from paddleocr import PaddleOCR

args = get_args()

class ExtractingText:
    def __init__(self, usingGPU: bool):
        self.gpu = usingGPU
        self.ocr = PaddleOCR(use_gpu=self.gpu)

    def cleanup_text(self, text):
        return " ".join([c if ord(c) < 128 else " " for c in text]).strip()

    def get_results(self, ocr_results, de_prob):
        words = []
        for (bbox, text, prob) in ocr_results:
            if prob > de_prob and len(text) > 2:
                text = self.cleanup_text(text)
                words.append(text)
        return words

    def PaddleOCR(self, dtset, languages, de_prob=0.5):
        char_pos = []
        for idx in dtset:
            img_file = os.path.join(args.data_path, os.path.basename(idx))
            img = cv2.imread(img_file)
            ocr_results = self.ocr.ocr(img)
            bboxes = self.get_results(ocr_results, de_prob)
            char_pos.append([idx, bboxes])
        return char_pos

'''