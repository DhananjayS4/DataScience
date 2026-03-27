from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class BlipPredictor:
    def __init__(self):
        # HARDCODE TO CPU for RTX 50-series (Blackwell) until Torch 2.7+ is stable 
        self.device = "cpu" 
        print(f"⚠️ Blackwell GPU (RTX 50xx) detected. Using CPU for stability...")
        
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        print(f"✅ BLIP Loaded on: {self.device}")

    def generate_caption(self, image_path):
        try:
            raw_image = Image.open(image_path).convert('RGB')
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=40)
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            return f"Error generating caption: {e}"
