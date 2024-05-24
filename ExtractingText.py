import cv2
import os
from config import get_args
from paddleocr import PaddleOCR

args = get_args()

class ExtractingText:
  def __init__(self, usingGPU: bool):
    self.gpu = usingGPU

  def cleanup_text(self, text):
    return " ".join([c if ord(c) < 128 else " " for c in text]).strip()
    
  def get_results(self, ocr_results, de_prob):
    words = []
    for (bbox, text, prob) in ocr_results:
      if prob > de_prob and len(text) > 2:
        text = self.cleanup_text(text)
        words.append(text)
    return words

  def PaddleOCR(self, dtset, languages, de_prob = 0.5):
    char_pos = []
    for idx in dtset:
        img_file = os.path.join(args.data_path, os.path.basename(idx))
        img = cv2.imread(img_file)
        ocr = PaddleOCR(use_gpu=self.gpu)
        results = ocr.ocr(img)
        bboxes = self.get_results(results, de_prob)
        char_pos.append([idx, bboxes])
    return char_pos
