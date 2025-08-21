from test_gradio import load_image, image_editing
from PIL import Image, ImageDraw
import numpy as np
from test_gradio import load_image
import torch
import options.options as option
from models import create_model as create_model_editguard
from app_utils import calculate_similarity_percentage


def image_model_select(ckp_index=0):
    print("Initialize model for watermarking, model index: ", ckp_index)
    
    if ckp_index != 0:
        print("Invalid model")
        return None
    
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

def hiding(image_input, bit_input, model):
    print("========== Image Embedding ==========")
    print("Input image shape: ", image_input.shape)
    print("Message", bit_input)
    
    message = np.array([int(bit_input[i:i+1]) for i in range(0, len(bit_input), 1)])
    message = message - 0.5
    val_data = load_image(image_input, message)
    model.feed_data(val_data)
    container = model.image_hiding()

    from PIL import Image
    image = Image.fromarray(container)
    return container, container

def ImageEdit(img, prompt, model_index):
    image, mask = img["image"], np.float32(img["mask"])

    received_image = image_editing(image, mask, prompt)
    return received_image, received_image, received_image

def revealing(image_edited, input_bit, model):
    number = 0.2

    container_data = load_image(image_edited) ## load tampered images
    model.feed_data(container_data)
    mask, remesg = model.image_recovery(number)
    mask = Image.fromarray(mask.astype(np.uint8))
    remesg = remesg.cpu().numpy()[0]
    remesg = ''.join([str(int(x)) for x in remesg])
    bit_acc = calculate_similarity_percentage(input_bit, remesg)
    return mask, remesg, bit_acc