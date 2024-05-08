import torch
import clip
from PIL import Image
import glob 
import numpy as np 



device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# print(clip.available_models())

# Function for embedding images
def Images(image):
    processed_image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embeddings = model.encode_image( processed_image)
    return image_embeddings


# Function for embedding text
def Text(text: str):
    text_tokens = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
    return text_embedding


# Function for comparing image and text similarity
def Compare(image, text: str):
    print(text)
    image = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(text).to(device)
    
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return np.ravel(probs)

# Testing with single image
# image = Image.open('guy.jpg')
# image = image.resize((224, 224))
# result = Images(image=image)
# print(result)    

# Testing the text embedding
# text_result = Text(text='coding is fun')
# print(text_result)
   

# test with multiple images from a folder
image_path = glob.glob('images/*.jpg')
for images in image_path:
    all_images = Image.open(images)
    all_images = all_images.resize((224,222))
    result = Images(image=all_images)
    print(result)



# similarity_result = Compare(image=image,text=['a dog','a person','a man','a cat'])
# print(similarity_result)

