import torch
from torchvision import transforms
from torchvision.models import resnet50
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

class ResNetEmbedding:
    def __init__(self, model_name="resnet50"):
        self.model = resnet50(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_embedding(self, img):
        img = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(img)
        return embedding.squeeze().cpu().numpy()