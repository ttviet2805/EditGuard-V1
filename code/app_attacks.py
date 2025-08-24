import torch
import numpy as np
from utils.JPEG import DiffJPEG
from utils.util import tensor2img, save_img
from PIL import Image
import app_utils
import cv2
from app_utils import rgb_to_bgr, bgr_to_rgb


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
    
    image_numpy = np.ascontiguousarray(image_numpy)
    image_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
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
        # # Turn image to RGB
        lr_img = app_utils.bgr_to_rgb(lr_img)
        
    print("Attacked image type, shape: ", type(lr_img), lr_img.shape)
    return lr_img, lr_img


# tmp = "/workspace/128x128_image_1.png"
# img_np = cv2.imread(tmp)
# save_img(img_np, "/workspace/origin_attack_image.png")

# attacked_img, _ = innoguard_attack(img_np, 0)
# save_img(attacked_img, "/workspace/attacked_image.png")