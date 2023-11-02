import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text_list = ["a diagram", "a dog", "a cat"]
text = clip.tokenize(text_list).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    image_features = image_features / image_features.norm(dim = -1, keep_dims = True)
    text_features = text_features / text_features.norm(dim = -1, keep_dims = True)
    
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

print("Top predictions:")
for value, index in zip(values, indices):
    print("%s: %.2f" % (text_list[index], 100 * value.item()))
