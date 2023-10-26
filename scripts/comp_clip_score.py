import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    image_features = image_features / image_features.norm(dim = -1, keep_dims = True)
    text_features = text_features / text_features.norm(dim = -1, keep_dims = True)

    clip_score = torch.matmul(image_features, text_features.T).item()

print("CLIP score:", clip_score)  # prints: 0.16
