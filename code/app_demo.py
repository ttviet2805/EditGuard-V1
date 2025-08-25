# ============================== Import my own lib ==============================
import gradio as gr
import app_utils
import backend
import app_attacks

# ----- VN START -----
import sys
import logging

# Open the file with UTF-8 encoding to support emojis and Unicode
logfile = open("app_console.log", "w", encoding="utf-8", buffering=1)

sys.stdout = logfile
sys.stderr = logfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=logfile
)
# ----- VN END -----


# Examples
examples = [
    ["../dataset/examples/0011.png"],
    ["../dataset/examples/0012.png"],
    ["../dataset/examples/0003.png"],
    ["../dataset/examples/0004.png"],
    ["../dataset/examples/0005.png"],
    ["../dataset/examples/0006.png"],
    ["../dataset/examples/0007.png"],
    ["../dataset/examples/0008.png"],
    ["../dataset/examples/0009.png"],
    ["../dataset/examples/0010.png"],
    ["../dataset/examples/0002.png"],
]

default_example = examples[0]


# =========================================================== FRONT END ===========================================================

logo_base64 = app_utils.img_to_base64("../logo.png")

html_content = f"""
<div style='display: flex; align-items: center; justify-content: center; padding: 20px;'>
    <img src='data:image/png;base64,{logo_base64}' alt='Logo' style='height: 50px; margin-right: 20px;'>
    <strong><font size='8'>EditGuard<font></strong>
</div>
"""

# Description
title = "<center><strong><font size='8'>EditGuard<font></strong></center>"

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

with gr.Blocks(css=css, title="EditGuard") as demo:
    gr.HTML(html_content)
    save_h = gr.State(value = None)
    save_w = gr.State(value = None)
    sam_global_points = gr.State([])
    sam_global_point_label = gr.State([])
    sam_original_image = gr.State(value=None)
    sam_mask = gr.State(value=None)
    
    # =========================================================== DATA INIT ===========================================================
    model = gr.State(value = backend.image_model_select(0))
    is_rgb_image = gr.State(value = True)
    image_edit_value = gr.State(value = None)
    image_edited_1_value = gr.State(value = None)

    with gr.Tabs():
        with gr.TabItem('Multipurpose Forensic Watermark'):

            DESCRIPTION = """
            ## How to use:
            - Upload an image and a copyright watermark (64-bit bitstring), then click "Embed Watermark" to generate a watermarked image.
            - Paint over the region to edit, and use the inpainting algorithm to edit the image.
            - Click "Extract" to detect tampered regions and output the copyright watermark."""
            
            gr.Markdown(DESCRIPTION)
            save_inpainted_image = gr.State(value=None)
            with gr.Column():
                with gr.Group():
                    gr.Markdown("# 1. Embed Watermark")
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(sources='upload', label="Original Image", interactive=True, type="numpy")
                            with gr.Row():
                                bit_input = gr.Textbox(label="Enter Copyright Watermark (64-bit bitstring)", placeholder="Enter here...")
                                rand_bit = gr.Button("🎲 Generate Random Watermark")
                            hiding_button = gr.Button("Embed Watermark")
                        with gr.Column():
                            image_watermark = gr.Image(label="Image with Watermark", interactive=False, type="numpy")


                with gr.Group():
                    gr.Markdown("# 2. Tamper the Image")
                    with gr.Row():
                        with gr.Column():
                            image_edit = gr.Image(sources='upload', label="Select Tampered Region", interactive=True, type="numpy")
                            attacking_model_list = gr.Dropdown(label="Select Tampering Model", choices=["Model 1: JPEG 70",  "Model 2: Gaussian 10", "Model 3: SD_inpainting"], type = 'index')
                            # text_prompt = gr.Textbox(label="Tampering Prompt")
                            attacking_button = gr.Button("Tamper Image")
                        with gr.Column():
                            image_edited = gr.Image(label="Tampered Result", interactive=False, type="numpy")
                

                with gr.Group():
                    gr.Markdown("# 3. Extract Watermark & Tampered Region")
                    with gr.Row():
                        with gr.Column():
                            image_edited_1 = gr.Image(sources="upload", label="Image to Extract From", interactive=True, type="numpy")
                            
                            revealing_button = gr.Button("Extract")
                        with gr.Column():
                            edit_mask = gr.Image(sources='upload', label="Edit Region Mask Prediction", interactive=True, type="numpy")
                            bit_output = gr.Textbox(label="Predicted Copyright Watermark")
                            acc_output = gr.Textbox(label="Watermark Prediction Accuracy")
                
                gr.Examples(
                    examples=examples,
                    inputs=[image_input],
                )

                # Embed Watermark
                hiding_button.click(
                    backend.hiding, inputs=[image_input, bit_input, model, is_rgb_image], outputs=[image_watermark, image_edit, image_edit_value]
                )
                rand_bit.click(
                    app_utils.rand, inputs=[], outputs=[bit_input]
                )

                # Tamper Image
                attacking_button.click(
                    app_attacks.innoguard_attack, inputs = [image_edit_value, attacking_model_list, is_rgb_image], outputs=[image_edited, image_edited_1, image_edited_1_value]
                )

                # Extract Watermark
                revealing_button.click(
                    backend.revealing, inputs=[image_edited_1, bit_input, model, is_rgb_image], outputs=[bit_output, acc_output]
                )

# Deploy server
demo.launch(server_name="0.0.0.0", server_port=2002, share=True, favicon_path='../logo.png')
# demo.launch(server_name="127.0.0.1", server_port=2002, share=False, favicon_path='../logo.png')


# ==================================================================================================================
# from PIL import Image
# import numpy as np
# import cv2
# from utils.util import save_img, tensor2img

# model = backend.image_model_select(0)


# input_path = "/workspace/EditGuard-V1/dataset/valAGE-Set/0000.png"
# img_np = cv2.imread(input_path)
# input_bit = "001110010010010111110000111111"
            # "001100010100010110110001011111"
# save_img(img_np, "/workspace/ori_image.png")
# # Image.fromarray(img_np, mode="RGB").save("/workspace/ori_image_RGB.png")

# out, _, _ = backend.hiding(img_np, input_bit, model)
# save_img(out, "/workspace/container_image.png")

# print("Type image: ", type(img_np))
# print("Type out: ", type(out))


# attacked_img, _, _ = app_attacks.innoguard_attack(out, 0)
# save_img(attacked_img, "/workspace/attacked_image.png")


# # bit, bit_acc = backend.revealing(out, input_bit, model)
# bit, bit_acc = backend.revealing(attacked_img, input_bit, model)

# ==================================================================================================================
# from PIL import Image
# import numpy as np
# from utils.util import save_img, tensor2img

# model = backend.image_model_select(0)

# # --- Load with PIL (RGB) ---
# input_path = "/workspace/EditGuard-V1/dataset/valAGE-Set/0000.png"
# img_pil = Image.open(input_path).convert("RGB")
# img_np = np.array(img_pil)   # HWC, RGB, uint8

# input_bit = "001110010010010111110000111111"

# # Save original for sanity check
# Image.fromarray(img_np, mode="RGB").save("/workspace/ori_image.png")

# # Hide watermark
# out, out = backend.hiding(img_np, input_bit, model)
# Image.fromarray(out, mode="RGB").save("/workspace/container_image.png")

# print("Type image: ", type(img_np), img_np.shape)
# print("Type out: ", type(out), getattr(out, "shape", None))

# # Attack image
# attacked_img, _ = app_attacks.innoguard_attack(out, 0)
# Image.fromarray(attacked_img, mode="RGB").save("/workspace/attacked_image.png")

# # Reveal watermark
# bit, bit_acc = backend.revealing(out, input_bit, model)
# bit, bit_acc = backend.revealing(attacked_img, input_bit, model)

# ==================================================================================================================
# from PIL import Image
# import numpy as np
# import cv2
# from utils.util import save_img, tensor2img

# model = backend.image_model_select(0)


# input_path = "/workspace/1024x1024_image_1.png"
# img_input_np = cv2.imread(input_path)
# save_img(img_input_np, "/workspace/ori_image.png")
# metadata_input = "vietpro_123"
# type_ECC = 0

# out_image, embed_message = backend.innoguard_hiding(img_input_np, metadata_input, type_ECC, model)
# save_img(out_image, "/workspace/watermarking_image.png")

# out_message = backend.innoguard_revealing(out_image, type_ECC, model)

# bit_acc = app_utils.calculate_similarity_percentage(embed_message, out_message)
# print(bit_acc)

# out, _, _ = backend.hiding(img_np, input_bit, model)
# save_img(out, "/workspace/container_image.png")

# print("Type image: ", type(img_np))
# print("Type out: ", type(out))


# attacked_img, _, _ = app_attacks.innoguard_attack(out, 0)
# save_img(attacked_img, "/workspace/attacked_image.png")


# # bit, bit_acc = backend.revealing(out, input_bit, model)
# bit, bit_acc = backend.revealing(attacked_img, input_bit, model)