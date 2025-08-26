import gradio as gr
import backend
import json

def make_payload(plate: str):
    return plate

def render_tab(model):
    with gr.TabItem("🛵 Application 2: Camera-based traffic fines"):

        type_ECC = gr.State(value = None)
        is_rgb_image = gr.State(value = True)
        out_message = gr.State(value = None)
        out_bit = gr.State(value = None)
        embed_status = gr.State(value = "")
        download_watermark_image = gr.State(value = "")

        DESCRIPTION = """Register license plate into the image"""
        gr.Markdown(DESCRIPTION)
        with gr.Group():
            gr.Markdown("## 1. Registration Phase")
            with gr.Row():
                with gr.Column(scale=1, min_width=200):  
                    with gr.Row():
                        image_input = gr.Image(sources="upload", label="Original Image",
                                            interactive=True, type="numpy")

                    gr.HTML("<div style='height:10px;'></div>")

                    with gr.Row():
                        license_plate_input = gr.Textbox(label="Enter the license plate", placeholder="Enter here...")

                with gr.Column(scale=1, min_width=200):
                    image_watermark = gr.Image(label="Watermarked image", interactive=False, show_download_button=False)
                    gr.HTML("<div style='height:20px;'></div>")
                    embed_btn = gr.Button("➡️ Embed into image")
                    download_watermark_image = gr.File(label="Download Embed Image PNG")
                    embed_status = gr.Textbox(label="Embed Status", interactive=False)

                # --------- embed button click here -----------
                embed_btn.click(
                    lambda img, lp, ecc, m, rgb: backend.innoguard_hiding(
                        img,
                        make_payload(lp),   # convert string to dict
                        ecc,
                        m,
                        rgb
                    ),
                    inputs = [image_input, license_plate_input, type_ECC, model, is_rgb_image],
                    outputs = [image_watermark, out_message, embed_status, download_watermark_image]
                )

        with gr.Group():
            gr.Markdown("## 2. Extraction Phase")
            with gr.Row():        
                with gr.Column():
                    image_rec = gr.Image(label="Distorted image", interactive=True, sources="upload", type="numpy")
                    gr.HTML("<div style='height:20px;'></div>")
                    extract_btn = gr.Button("➡️ Extract information from image")

                    # --------- extract button click here ---------

                with gr.Column():
                    extracted_output = gr.Textbox(label="Output license plate", lines=10, interactive=False)

                extract_btn.click(
                    backend.innoguard_revealing,
                    inputs = [image_rec, type_ECC, model, is_rgb_image],
                    outputs = [out_bit, extracted_output]
                )