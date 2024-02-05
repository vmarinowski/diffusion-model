import kaggle
import torch
import torch.nn as nn
import PIL
import torchvision.transforms as transforms
import numpy as np
import os

# You need a kaggle account for downloading the dataset
dataset = 'rick-and-morty'

!kaggle datasets download -d dataset
# {"username": "<userID.", "key": "<userKey>"}

!unzip titanic.zip

image_shape = (128, 128)
batch_size = 128

transform_img = transforms.Compose([
    transforms.Resize(image_shape),
    #transforms.RandomHorizontalFlip(p = 0.3), -> This transformation is optional
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

retransform_img = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),
    transforms.Lambda(lambda t: t.permute(1,2,0)),
    transforms.Lambda(lambda t: t * 255),
    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
    transforms.ToPILImage()
])

dir_path = "path to your image directory"

def get_images(path_to_img = dir_path):
    images = []
    for f in os.scandir('/kaggle/input/rick-and-morty/rick_and_morty'):
        for step, image_dir in enumerate(os.scandir(f.path)):
            image_path = image_dir.path
            image = PIL.Image.open(image_path)
            images.append(transform_img(image))
    
    return torch.stack(images, dim = 0)

images = get_images()

dataloader = torch.utils.data.DataLoader(images, batch_size = batch_size, drop_last = True)