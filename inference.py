# inference.py
import torch
from transformers import BertTokenizer

class ModelInference:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = torch.load(model_path)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        return predictions.item()
