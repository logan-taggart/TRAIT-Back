import torch
from transformers import BeitImageProcessor, BeitModel, CLIPModel, CLIPProcessor

class BEiTEmbedding:
    def __init__(self, model_name="microsoft/beit-base-patch16-224"):
        self.feature_extractor = BeitImageProcessor.from_pretrained(model_name) 
        self.model = BeitModel.from_pretrained(model_name)

    def extract_embedding(self, img):
        inputs = self.feature_extractor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy() 
        return embedding

class CLIPEmbedding:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def extract_embedding(self, img):
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs.squeeze().numpy()