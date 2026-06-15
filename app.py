import gradio as gr
import numpy as np

def analyze_image(image):
    return "Analysis complete."

def clear_fields():
    return None, ""

# Minimalist structure
with gr.Blocks() as demo:
    img_in = gr.Image()
    out = gr.Textbox()
    btn = gr.Button("Analyze")
    btn.click(analyze_image, inputs=[img_in], outputs=[out])

# Launch with absolute minimal interference
if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
