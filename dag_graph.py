from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from logger import log_event

class InferenceNode:
    def __init__(self, model_path="model_output"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Device set to use {self.device}")

    def run(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, label = torch.max(probs, dim=1)
        confidence = confidence.item()
        label = "Positive" if label.item() == 1 else "Negative"
        log_event("Inference", f"Predicted: {label} | Confidence: {confidence:.2f}")
        return label, confidence

class ConfidenceCheckNode:
    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def run(self, confidence):
        if confidence < self.threshold:
            log_event("ConfidenceCheck", "Low confidence detected. Triggering fallback.")
            return False
        return True

class FallbackNode:
    def run(self, original_text):
        print("System: Could you clarify your intent? Was this a positive or negative review?")
        user_input = input("User: ").strip().lower()
        if "neg" in user_input:
            label = "Negative"
        elif "pos" in user_input:
            label = "Positive"
        else:
            label = "Uncertain"
        log_event("Fallback", f"User clarified to: {label}")
        return label