import gradio as gr
import backend
import json

def make_payload(plate: str):
    return {"License Plate": plate}

def render_tab(model):
    with gr.TabItem("📰 Application 3: News/Articles images protection"):

        type_ECC = gr.State(value = None)
        is_rgb_image = gr.State(value = True)
        out_message = gr.State(value = None)
        out_bit = gr.State(value = None)
        # json_input = gr.State(value = None)
        embed_status = gr.State(value = "")
        download_watermark_image = gr.State(value = None)

        DESCRIPTION = """Register news/articles's information into the image"""
        gr.Markdown(DESCRIPTION)
        with gr.Group():
            gr.Markdown("## 1. Registration Phase")
            with gr.Row():
                with gr.Column(scale=1, min_width=200):  
                    with gr.Row():
                        image_input = gr.Image(sources="upload", label="Original Image",
                                            interactive=True, type="numpy")

                    gr.HTML("<div style='height:10px;'></div>")

                    source_input = gr.Textbox(label="Enter the source", placeholder="Enter here...")
                    url_input = gr.Textbox(label="Enter the URL", placeholder="Enter here...")
                    date_input = gr.Textbox(label="Enter the date of publication", placeholder="Enter here...")
                    title = gr.Textbox(label="Enter the title", placeholder="Enter here...")
                    category = gr.Textbox(label="Enter the category", placeholder="Enter here...")
                    
                    json_input = gr.JSON(label="Collected data")

                    with gr.Row():
                        with gr.Column():
                            collect_btn = gr.Button("Collect", variant="primary")
                            embed_btn = gr.Button("➡️ Embed into image", variant = "secondary")

                with gr.Column(scale=1, min_width=200):
                    image_watermark = gr.Image(label="Watermarked image", interactive=False, show_download_button=False)
                    gr.HTML("<div style='height:10px;'></div>")
                    embed_status = gr.Textbox(label="Embed Status", interactive=False)
                    download_watermark_image = gr.File(label="Download Watermarked Image PNG") 

                # --------- embed button click here -----------
                def on_collect(src, url, date, title, category):
                    metadata = {}
                    if src: 
                        metadata["source"] = src.strip()
                    if url: 
                        metadata["url"] = url.strip()
                    if date: 
                        metadata["date"] = date.strip()
                    if title: 
                        metadata["title"] = title.strip()
                    if category: 
                        metadata["category"] = category.strip()
                    return {"Data": metadata}

                collect_btn.click(
                    on_collect,
                    inputs =  [source_input, url_input, date_input, title, category],
                    outputs = [json_input]
                )

                embed_btn.click(
                    backend.innoguard_hiding,
                    inputs = [image_input, json_input, type_ECC, model, is_rgb_image],
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
                    extracted_output = gr.Textbox(label="Output data", lines=10, interactive=False)

                extract_btn.click(
                    backend.innoguard_revealing,
                    inputs = [image_rec, type_ECC, model, is_rgb_image],
                    outputs = [out_bit, extracted_output]
                )