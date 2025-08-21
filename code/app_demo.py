# ============================== Import my own lib ==============================
import gradio as gr
import app_utils
import backend

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
    model = gr.State(value = None)
    save_h = gr.State(value = None)
    save_w = gr.State(value = None)
    sam_global_points = gr.State([])
    sam_global_point_label = gr.State([])
    sam_original_image = gr.State(value=None)
    sam_mask = gr.State(value=None)

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
                with gr.Row():
                    model_list = gr.Dropdown(label="Select Model", choices=["Model 1"], type = 'index')
                    clear_button = gr.Button("Clear All")
                with gr.Group():
                    gr.Markdown("# 1. Embed Watermark")
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(sources='upload', label="Original Image", interactive=True, type="numpy", value=default_example[0])
                            with gr.Row():
                                bit_input = gr.Textbox(label="Enter Copyright Watermark (64-bit bitstring)", placeholder="Enter here...")
                                rand_bit = gr.Button("🎲 Generate Random Watermark")
                            hiding_button = gr.Button("Embed Watermark")
                        with gr.Column():
                            image_watermark = gr.Image(sources="upload", label="Image with Watermark", interactive=True, type="numpy")


                with gr.Group():
                    gr.Markdown("# 2. Tamper the Image")
                    with gr.Row():
                        with gr.Column():
                            image_edit = gr.Image(sources='upload',image_mode="sketch", label="Select Tampered Region", interactive=True, type="numpy")
                            inpainting_model_list = gr.Dropdown(label="Select Tampering Model", choices=["Model 1: SD_inpainting"], type = 'index')
                            text_prompt = gr.Textbox(label="Tampering Prompt")
                            inpainting_button = gr.Button("Tamper Image")
                        with gr.Column():
                            image_edited = gr.Image(sources="upload", label="Tampered Result", interactive=True, type="numpy")
                

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

                # Loading model
                model_list.change(
                    backend.image_model_select, inputs = [model_list], outputs=[model]
                )

                # Embed Watermark
                hiding_button.click(
                    backend.hiding, inputs=[image_input, bit_input, model], outputs=[image_watermark, image_edit]
                )
                rand_bit.click(
                    app_utils.rand, inputs=[], outputs=[bit_input]
                )

                # Tamper Image
                inpainting_button.click(
                    backend.ImageEdit, inputs = [image_edit, text_prompt, inpainting_model_list], outputs=[image_edited, image_edited_1, save_inpainted_image]
                )

                # Extract Watermark
                revealing_button.click(
                    backend.revealing, inputs=[image_edited_1, bit_input, model_list, model], outputs=[edit_mask, bit_output, acc_output]
                )

# Deploy server
demo.launch(server_name="0.0.0.0", server_port=2002, share=True, favicon_path='../logo.png')