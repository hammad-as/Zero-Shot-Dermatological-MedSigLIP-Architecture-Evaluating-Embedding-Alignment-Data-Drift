import os
import logging
import gradio as gr
import numpy as np

# 1. MLOps: Setup Logging
logging.basicConfig(level=logging.INFO, filename='inference_logs.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_inference(diagnosis, confidence):
    logging.info(f"Prediction: {diagnosis} | Confidence: {confidence}")

def analyze_image(image):
    if image is None:
        return "⚠️ No image detected.", "0%", "N/A"
    
    # Simulation logic
    diagnosis = "Melanocytic Nevus"
    confidence = "92.45%"
    
    log_inference(diagnosis, confidence)
    return f"🔬 Analysis Complete\nCondition: {diagnosis}", confidence, diagnosis

def clear_fields():
    return None, "", "", ""

# 2. UI: Standard Blocks setup (Removed 'api_open')
with gr.Blocks(title="MedSigLIP Dashboard") as demo:
    gr.Markdown("# 🔬 MedSigLIP Diagnostic Dashboard")
    
    with gr.Row():
        img_in = gr.Image(type="numpy", label="Upload Dermatological Imagery")
        with gr.Column():
            logs_out = gr.Textbox(label="Status")
            score_out = gr.Textbox(label="Confidence")
            cond_out = gr.Textbox(label="Condition")
    
    with gr.Row():
        analyze_btn = gr.Button("Analyze Image", variant="primary")
        clear_btn = gr.Button("Clear")

    analyze_btn.click(
        fn=analyze_image, 
        inputs=[img_in], 
        outputs=[logs_out, score_out, cond_out]
    )
    clear_btn.click(
        fn=clear_fields, 
        inputs=None, 
        outputs=[img_in, logs_out, score_out, cond_out]
    )

if __name__ == "__main__":
    # Disable API and analytics in the launch function instead of the class constructor
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        show_api=False
    )
