import numpy as np
import torch
import clip
from torchvision import transforms


def clipper(images, prompt):
    "ranks images based on CLIP score"
    "returns a list of images and a list of scores"

    # transform tensors for CLIP
    crop = transforms.CenterCrop(size=(224, 224))
    norm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711))
    timgs = []
    for img in images:
        timg = crop(img)
        timg = norm(timg)
        timgs.append(timg)
    
    # load CLIP
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()

    # tokenize text and extract features
    text_tokens = clip.tokenize("This is " + prompt).cuda()
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # extract image features
    image_features = []
    for timg in timgs:
        with torch.no_grad():
            image_features.append(model.encode_image(timg[None, :].cuda()).float())
    
    # compute cosine singularity for each image
    similarity = []
    for i in range(len(image_features)):
        image_features[i] /= image_features[i].norm(dim=-1, keepdim=True)
        text_features.cpu().detach().numpy()
        similarity.append(text_features.cpu().detach().numpy() @ image_features[i].cpu().numpy().T)

    output = list(zip(images, similarity))
    output = sorted(output, key=lambda x: x[1])[::-1]
    
    ims = [row[0] for row in output]
    sims = [row[1] for row in output]

    return ims, sims