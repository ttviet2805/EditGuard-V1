import torch
import numpy as np
from utils.JPEG import DiffJPEG
from utils.util import tensor2img
from PIL import Image


def JPEG_image_degradation(image, NL):
    image = image.astype(np.float32)
    image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    image = image.unsqueeze(0)
    JPEG = DiffJPEG(differentiable=True, quality=int(NL))
    y_forw = JPEG(image)
    y_forw = y_forw.permute(0, 2, 3, 1)
    y_forw = y_forw.cpu().detach().numpy().squeeze()
    y_forw = (y_forw * 255.0).astype(np.uint8)

    return y_forw, y_forw

def Gaussian_image_degradation(image, NL):
    image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    image = image.unsqueeze(0)
    NL = NL / 255.0
    noise = np.random.normal(0, NL, image.shape)
    torchnoise = torch.from_numpy(noise).float()
    y_forw = image + torchnoise
    y_forw = torch.clamp(y_forw, 0, 1)
    y_forw = y_forw.permute(0, 2, 3, 1)
    y_forw = y_forw.cpu().detach().numpy().squeeze()

    y_forw = (y_forw * 255.0).astype(np.uint8)
    return y_forw, y_forw

def innoguard_attack(image_numpy, attack_type):
    print("========== Image Attacking ==========")
    print("Attack type: ", attack_type)
    print("Image type, shape", type(image_numpy), image_numpy.shape)
    
    image_numpy = image_numpy.astype(np.float32)
    image_numpy = torch.from_numpy(np.transpose(image_numpy, (2, 0, 1)))
    image_tensor = image_numpy.unsqueeze(0)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    y_forw = image_tensor.to(device)
    
    print("y_forw type, shape", type(y_forw), y_forw.shape, "device:", y_forw.device)
    
    with torch.no_grad():
        if (attack_type == 0): # JPEG Compression
            NL = 70
            print("Attack name: JPEG ", NL)
            diffJPEG = DiffJPEG(differentiable=True, quality=int(NL)).to(device)
            y_forw = diffJPEG(y_forw)
            
        elif (attack_type == 1): # Gaussian Noise
            NL = 10 / 255.0
            print("Attack name: Gaussian Noise ", NL * 255.0)
            noise = np.random.normal(0, NL, y_forw.shape)
            torchnoise = torch.from_numpy(noise).float().to(device)
            y_forw = y_forw + torchnoise

        
        result = torch.clamp(y_forw,0,1)
        lr_img = tensor2img(result)

    print("LR image type, shape: ", type(lr_img), lr_img.shape)

    return lr_img, lr_img

tmp = "/workspace/128x128_image_1.png"
innoguard_attack(np.array(Image.open(tmp).convert('RGB')), 0)