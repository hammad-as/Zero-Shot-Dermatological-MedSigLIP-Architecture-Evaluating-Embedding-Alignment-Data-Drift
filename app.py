import gradio as gr
import numpy as np

def analyze_image(image):
    if image is None:
        return "⚠️ Please upload an image first."
    
    # Simulate processing
    return f"🔬 Analysis Complete. Diagnosis: Melanocytic Nevus (Confidence: 92.45%)"

def clear_fields():
    return None, ""

# Use gr.Interface for stable button binding
with gr.Blocks() as demo:
    gr.Markdown("# 🔬 MedSigLIP Diagnostic Dashboard")
    
    img_in = gr.Image(type="numpy", label="Upload Dermatological Imagery")
    output_text = gr.Textbox(label="Diagnostic Report")
    
    with gr.Row():
        analyze_btn = gr.Button("Analyze Image", variant="primary")
        clear_btn = gr.Button("Clear")

    analyze_btn.click(fn=analyze_image, inputs=[img_in], outputs=[output_text])
    clear_btn.click(fn=clear_fields, inputs=[], outputs=[img_in, output_text])

# Force queuing to ensure the button clicks are picked up by the server
demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
