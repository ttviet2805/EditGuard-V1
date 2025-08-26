from test_gradio import load_image, image_editing
import time
from PIL import Image, ImageDraw
import numpy as np
from test_gradio import load_image
import torch
import options.options as option
from models import create_model as create_model_editguard
from app_utils import calculate_similarity_percentage, rgb_to_bgr, bgr_to_rgb
import app_utils
from utils.util import calculate_psnr
import json

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
    # print(image_input)
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
    # print(image_edited)
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

def revealing_no_accuracy_calculation(image_edited, model, is_rgb_image = False):
    if is_rgb_image == True:
        image_edited = rgb_to_bgr(image_edited)
    
    print("========== Image Extracting ==========")
    print("Extracted image type, shape: ", type(image_edited), image_edited.shape)
    # print(image_edited)
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
    print("Receive message: ", remesg)
    return remesg

def innoguard_hiding(image_input, metadata_input, type_ECC, model, is_rgb_image = False):
    print("================================================== InnoGuard Image Embedding ==================================================")
    print("Input image type, shape: ", type(image_input), image_input.shape)
    # print(image_input)
    print("Message: ", metadata_input)
    
    if (type(metadata_input) == dict):
        metadata_input = json.dumps(metadata_input)
        print("Message after dump: ", metadata_input)
    # Constant
    SUB_IMAGE_SIZE = 128
    SUB_IMAGE_BIT = 30
    
    # Input init
    metadata_input_bit = app_utils.encode_ascii(metadata_input)
    print("Message bit: ", metadata_input_bit)
    metadata_list, metadata_padding = app_utils.split_bits_30(metadata_input_bit)
    tiles_128, coords, orig_hw, padded_hw = app_utils.split_into_tiles_128(image_input, pad_mode="edge")
    num_child_images = len(tiles_128)
    list_container_numpy = []
    H, W, C = image_input.shape
    num_child_on_width_size, num_child_on_height_size = H//SUB_IMAGE_SIZE, W//SUB_IMAGE_SIZE
    max_characters = num_child_images * SUB_IMAGE_BIT // 8
    print("Maximum bit number: ", num_child_images * SUB_IMAGE_BIT)
    print("Maximum character number: ", max_characters)
    
    out_message = ""
    embed_status = "Data have been embedded into the image successfully."
    download_path = None
    
    if len(metadata_input) > max_characters:
        embed_status = f"Error: Message too long. Max characters allowed: {max_characters}. Current length: {len(metadata_input)}"
        return None, out_message, embed_status, download_path
    
    for i in range(0, num_child_images):
        if i < len(metadata_list):
            message = metadata_list[i]
        else:
            message = "0" * SUB_IMAGE_BIT
            
        out_message += message
    
        current_image_np, _, _ = hiding(tiles_128[i], message, model, is_rgb_image)
        list_container_numpy.append(current_image_np)
        
    parent_container = app_utils.combine_tiles_ordered(list_container_numpy, num_child_on_width_size, num_child_on_height_size)
    
    # print("PARENT CONTAINER: ", type(parent_container), parent_container.shape, '\n', parent_container)

    # Save image into temp folder
    download_path = "download_image.png"
    img = Image.fromarray(parent_container)
    img.save(download_path)
    
    return parent_container, out_message, embed_status, download_path

def innoguard_revealing(image_edited, type_ECC, model, is_rgb_image = False):
    print("================================================== InnoGuard Image Extracting ==================================================")
    print("Input image type, shape: ", type(image_edited), image_edited.shape)
    # print(image_edited)
    
    # Constant
    SUB_IMAGE_SIZE = 128
    SUB_IMAGE_BIT = 30
    
    # Input init
    H, W, C = image_edited.shape
    num_child_on_width_size, num_child_on_height_size = H//SUB_IMAGE_SIZE, W//SUB_IMAGE_SIZE
    num_child_images = num_child_on_width_size * num_child_on_height_size
    tiles_128, coords, orig_hw, padded_hw = app_utils.split_into_tiles_128(image_edited, pad_mode="edge")    
    num_child_images = len(tiles_128)
    print("Maximum bit number: ", num_child_images * SUB_IMAGE_BIT)
    
    out_bit = ""
    
    for i in range(0, num_child_images):
        message = revealing_no_accuracy_calculation(tiles_128[i], model, is_rgb_image)
        out_bit += message
    
    out_metadata = app_utils.decode_ascii_until_zero_byte(out_bit)
    print("Output metadata bit: ", out_bit)
    print("Output metadata: ", out_metadata)
    
    return out_bit, out_metadata