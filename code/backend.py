from test_gradio import load_image, image_editing
import time
from PIL import Image, ImageDraw
import numpy as np
from test_gradio import load_image
import torch
import options.options as option
from models import create_model as create_model_editguard
from app_utils import calculate_similarity_percentage, rgb_to_bgr, bgr_to_rgb
from utils.util import calculate_psnr


def image_model_select(ckp_index=0):
    print("Initialize model for watermarking, model index: ", ckp_index)
    
    if ckp_index != 0:
        raise ValueError("error message")
    
    # options
    opt = option.parse("options/test_editguard.yml", is_train=True)
    # distributed training settings
    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                    map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    torch.backends.cudnn.benchmark = True
    # create model

    model = create_model_editguard(opt)
    model_pth = '../checkpoints/16000_G.pth'
    print(model_pth)
    model.load_test(model_pth)
    return model

def hiding(image_input, bit_input, model, is_rgb_image = False):
    if is_rgb_image == True:
        image_input = rgb_to_bgr(image_input)
    
    print("========== Image Embedding ==========")
    print("Input image type, shape: ", type(image_input), image_input.shape)
    print(image_input)
    print("Message", bit_input)
    
    # from utils.util import save_img
    # save_img(image_input, "/workspace/image_input_cv2.png")
    # from PIL import Image
    # Image.fromarray(image_input, mode="RGB").save("/workspace/image_input_PIL.png")
    
    message = np.array([int(bit_input[i:i+1]) for i in range(0, len(bit_input), 1)])
    message = message - 0.5
    val_data = load_image(image_input, message)
    
    # --- measure feed_data ---
    t0 = time.perf_counter()
    model.feed_data(val_data)
    t1 = time.perf_counter()
    feed_ms = (t1 - t0) * 1000.0
    print(f"feed_data time: {feed_ms:.2f} ms")
    
    # --- measure image_hiding ---
    t2 = time.perf_counter()
    container = model.image_hiding()
    t3 = time.perf_counter()
    hiding_ms = (t3 - t2) * 1000.0
    total_ms  = (t3 - t0) * 1000.0
    print(f"image_hiding time: {hiding_ms:.2f} ms")
    print(f"Total (feed_data + image_hiding): {total_ms:.2f} ms")
    
    print("PSNR: ", calculate_psnr(image_input, container))
    print("Container type, shape: ", type(container), container.shape)

    from PIL import Image
    image = Image.fromarray(container)
    
    if is_rgb_image == True:
        rgb_container = bgr_to_rgb(container)
        return rgb_container.copy(), rgb_container.copy(), rgb_container.copy()
    else:
        return container.copy(), container.copy(), container.copy()

def ImageEdit(img, prompt, model_index):
    image, mask = img["image"], np.float32(img["mask"])

    received_image = image_editing(image, mask, prompt)
    return received_image, received_image, received_image

def revealing(image_edited, input_bit, model, is_rgb_image = False):
    if is_rgb_image == True:
        image_edited = rgb_to_bgr(image_edited)
    
    print("========== Image Extracting ==========")
    print("Extracted image type, shape: ", type(image_edited), image_edited.shape)
    print(image_edited)
    print("Message: ", input_bit)
    number = 0.2

    container_data = load_image(image_edited) ## load tampered images
    
    # ---- time: feed_data ----
    t0 = time.perf_counter()
    model.feed_data(container_data)
    t1 = time.perf_counter()
    feed_ms = (t1 - t0) * 1000.0
    print(f"feed_data time: {feed_ms:.2f} ms")
    
    # ---- time: image_recovery ----
    t2 = time.perf_counter()
    remesg = model.image_recovery(number)
    t3 = time.perf_counter()
    recovery_ms = (t3 - t2) * 1000.0
    total_ms = (t3 - t0) * 1000.0
    print(f"image_recovery time: {recovery_ms:.2f} ms")
    print(f"Total (feed_data + image_recovery): {total_ms:.2f} ms")
    
    remesg = remesg.cpu().numpy()[0]
    remesg = ''.join([str(int(x)) for x in remesg])
    bit_acc = calculate_similarity_percentage(input_bit, remesg)
    print("Receive message: ", remesg)
    print("Bit Accuracy: ", bit_acc)
    return remesg, bit_acc