import gradio as gr

# CUSTOM_CSS = """
# .circle-btn {
#     width: 60px !important;
#     height: 60px !important;
#     border-radius: 50% !important;
#     border: none !important;
#     background-color: #e5e7eb !important;
#     color: #111827 !important;
#     font-size: 18px !important;
#     font-weight: 600 !important;
#     cursor: pointer !important;
#     display: flex !important;
#     align-items: center !important;
#     justify-content: center !important;
#     box-shadow: 0 2px 4px rgba(0,0,0,0.2);
# }

# /* make the middle column flexbox & full height */
# .embed-col {
#     display: flex !important;
#     align-items: center !important;    /* vertical centering */
#     justify-content: center !important;/* horizontal centering */
#     height: 100% !important;
# }
# """

CUSTOM_CSS = """
/* Xóa nền xám của các component box */
.svelte-1ipelgc, 
.svelte-1ipelgc textarea,
.svelte-1ipelgc input,
.svelte-1ipelgc .wrap {
    background-color: white !important;
}

/* Riêng JSON component */
.json-wrap, .json-wrap pre {
    background-color: white !important;
}

/* Nếu còn Column bị màu xám */
.gradio-container .gr-block.gr-column {
    background: transparent !important;
}
"""

def render_tab():
    with gr.TabItem("👲 Application 3: ......."):
        DESCRIPTION = """Register each person’s name and its corresponding bounding box into the image"""
        gr.Markdown(DESCRIPTION)
        with gr.Group():
            gr.Markdown("## 1. Registration Phase")
            with gr.Row():
                with gr.Column(scale=2, min_width=500):  
                    with gr.Row():
                        image_input = gr.Image(sources="upload", label="Original Image",
                                            interactive=True, type="numpy")

                    gr.Markdown("#### Input names & bounding boxes")

                    MAX_ROWS = 10
                    row_count = gr.State(1)

                    # Tạo sẵn các hàng: [Textbox name] + 4 Number cho bbox
                    name_boxes = []
                    x1_boxes, y1_boxes, x2_boxes, y2_boxes = [], [], [], []

                    row_containers = []
                    for i in range(MAX_ROWS):
                        with gr.Row(visible=(i == 0)) as row:
                            name = gr.Textbox(label=f"Person {i+1}", placeholder="Name")
                            x1 = gr.Number(label="x1", precision=0)
                            y1 = gr.Number(label="y1", precision=0)
                            x2 = gr.Number(label="x2", precision=0)
                            y2 = gr.Number(label="y2", precision=0)
                        row_containers.append(row)
                        name_boxes.append(name)
                        x1_boxes.append(x1)
                        y1_boxes.append(y1)
                        x2_boxes.append(x2)
                        y2_boxes.append(y2)

                    with gr.Row():
                        add_btn = gr.Button("➕ Add person", variant="secondary")
                        rem_btn = gr.Button("➖ Remove last", variant="secondary")
                        collect_btn = gr.Button("Collect", variant="primary")

                    json_input = gr.JSON(label="Collected {name, bbox}")

                    # ---------- helpers ----------
                    def _rows_visibility(count):
                        # trả về list gr.update cho từng Row theo count
                        return [gr.update(visible=(i < count)) for i in range(MAX_ROWS)]

                    def on_add(count):
                        count = int(count)
                        if count < MAX_ROWS:
                            count += 1
                        return [count, *_rows_visibility(count)]

                    def on_remove(count):
                        count = int(count)
                        if count > 1:
                            count -= 1
                        # cũng có thể clear ô của hàng bị ẩn nếu muốn
                        return [count, *_rows_visibility(count)]

                    def on_collect(count, *vals):
                        # vals = [all names] + [all x1] + [all y1] + [all x2] + [all y2]
                        count = int(count)
                        n = MAX_ROWS
                        names = list(vals[0:n])
                        x1s   = list(vals[n:2*n])
                        ys1   = list(vals[2*n:3*n])
                        x2s   = list(vals[3*n:4*n])
                        ys2   = list(vals[4*n:5*n])

                        out = []
                        for i in range(count):
                            name = (names[i] or "").strip()
                            if not name:
                                continue
                            try:
                                x1 = float(x1s[i]); y1 = float(ys1[i])
                                x2 = float(x2s[i]); y2 = float(ys2[i])
                                if x2 < x1: x1, x2 = x2, x1
                                if y2 < y1: y1, y2 = y2, y1
                                out.append({"name": name, "bbox": [x1, y1, x2, y2]})
                            except Exception:
                                # bỏ hàng chưa đủ số
                                continue
                        return out

                    # ---------- wiring ----------
                    add_btn.click(on_add, inputs=[row_count],
                                outputs=[row_count, *row_containers])
                    rem_btn.click(on_remove, inputs=[row_count],
                                outputs=[row_count, *row_containers])

                    collect_btn.click(
                        on_collect,
                        inputs=[row_count, *name_boxes, *x1_boxes, *y1_boxes, *x2_boxes, *y2_boxes],
                        outputs=json_input
                    )
            
                with gr.Column(scale=1, min_width=200):
                    image_watermark = gr.Image(label="Watermarked image", interactive=False)
                    embed_btn = gr.Button("➡️ Embed into image")

                    # --------- embed button here -----------

        with gr.Group():
            gr.Markdown("## 2. Extraction Phase")
            with gr.Row():        
                with gr.Column():
                    image_rec = gr.Image(label="Distorted image", interactive=True, sources="upload", type="numpy")
                    extract_btn = gr.Button("➡️ Extract information from image")

                    # --------- extract button here ---------

                with gr.Column():
                    json_output = gr.JSON(label="Output {name, bbox}")