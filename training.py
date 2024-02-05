import torch
import torch.nn as nn
import numpy as np
import PIL
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import math
from tqdm import tqdm
import numpy as np
from data import dataloader
from .model import *
from .precalculated import *

def sample(number_of_images = 4, rand_timesteps = np.random.randint(low = 0, high = 128, size = (10,), dtype = int)):
    fig, ax = plt.subplots(1, number_of_images, figsize = (number_of_images * 5, 5))
    for delta in range(number_of_images):

        ax[delta].imshow(retransform_img(res[rand_timesteps[delta]]))
        ax[delta].set_title(f'Image: {delta}')
        ax[delta].axis('off')

def get_noisy_image(image, t):
    #t is a 1d Tensor
    count = 0
    dummy_image = torch.randn_like(image[0].view(1, 3, image_shape[0], image_shape[0]))
    dummy_eta = torch.randn_like(image[0].view(1, 3, image_shape[0], image_shape[0])) # dimension -> 1, C, H, W
    for time in t:
        img = image[count].view(1,3,image_shape[0],image_shape[0])
        eta = torch.randn_like(img) #image dimension -> 1, C, H, W
        count = count + 1
        noisy_image = alpha_hat_sqrt[time].to(device) * img.to(device) + one_minus_alpha_hat_sqrt[time] * eta.to(device)
        dummy_image = torch.cat([dummy_image, noisy_image], dim = 0)
        dummy_eta = torch.cat([dummy_eta, eta], dim = 0)
    
    return dummy_image[1:].to(device), dummy_eta[1:].to(device)

model = SimpleUnet()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
EPOCHS = 5
loss_fn = nn.MSELoss()
T = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for epoch in range(EPOCHS):
    
    for step, batch in enumerate(tqdm(dataloader, desc = f'Epoch: {epoch}')):
        
        optimizer.zero_grad()
        t_step = torch.randint(0, 1000, (batch_size,), device = device).long()
        imgs, noise = get_noisy_image(batch.to(device), t_step)
        predicted_noise = model(imgs, t_step)
        loss = loss_fn(predicted_noise, noise)
        #print(loss.item())
        loss.backward()
        optimizer.step()
        if step == 55:
            break
    
    print(f'Current loss: {loss}\n')

    with torch.no_grad():
        fig, ax = plt.subplots(1, 2, figsize = (3*2, 2))

        ax[0].imshow(retransform_img(predicted_noise[1,:,:,:].squeeze()))
        ax[1].imshow(retransform_img(noise[1, :, :, :].squeeze()))

        ax[0].set_title('Predicted noise')
        ax[1].set_title('Actual noise')
        
        plt.show()

#Sampling

model.eval()
with torch.no_grad():
    def get_sample(image, time):
        #t is a 1d Tensor
        count = 0
        dummy_image = torch.randn_like(image[0].view(1, 3, image_shape[0], image_shape[0]))
        dummy_eta = torch.randn_like(image[0].view(1, 3, image_shape[0], image_shape[0])) # dimension -> 1, C, H, W
        output = model(image, time)
        for t in time:
            img = image[count].view(1,3,image_shape[0],image_shape[0])
            eta = torch.randn_like(img) #image dimension -> 1, C, H, W
            model_mean = sqrt_recip_alpha[t] * (img - beta[t] * output[count] / alpha_hat_sqrt[t])
            count = count + 1
            ilike = model_mean + torch.sqrt(posterior_variance[t]) * eta
            dummy_image = torch.cat([dummy_image, ilike], dim = 0)
        return dummy_image[1:].to(device)

    for step, batch in enumerate(dataloader):
            if step == 70:
                test_batch = batch
                break

    res = get_sample(test_batch.to(device), 
                     torch.randint(0, T, (batch_size,), device = device).long())