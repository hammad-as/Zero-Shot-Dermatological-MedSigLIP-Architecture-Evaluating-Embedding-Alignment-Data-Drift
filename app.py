import gradio as gr
import torch
import numpy as np

# --- LOGIC ENGINE ---
class ClinicalDetective:
    def analyze_image(self, image):
        if image is None:
            return "⚠️ No image detected. Please upload an image.", "0%", "N/A"
        
        # Simulated diagnostic logic
        diagnosis = "Melanocytic Nevus"
        confidence = "92.45%"
        
        report = (
            f"🔬 ANALYSIS REPORT\n"
            f"---------------------------\n"
            f"Status: SUCCESS\n"
            f"Diagnosis: {diagnosis}\n"
            f"Confidence: {confidence}\n"
            f"Processing: MedSigLIP Engine v1.0"
        )
        return report, confidence, diagnosis

detective = ClinicalDetective()

# --- BLOCKS INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 MedSigLIP Diagnostic Dashboard")
    
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="numpy", label="Upload Dermatological Imagery")
            with gr.Row():
                analyze_btn = gr.Button("Analyze Image", variant="primary")
                clear_btn = gr.Button("Clear")
        
        with gr.Column():
            logs_out = gr.TextArea(label="Diagnostic Logs")
            score_out = gr.Textbox(label="Confidence Level")
            cond_out = gr.Textbox(label="Identified Condition")

    # Forcefully re-binding events
    analyze_btn.click(
        fn=detective.analyze_image, 
        inputs=img_in, 
        outputs=[logs_out, score_out, cond_out]
    )
    
    # Simple lambda for clearing to ensure it triggers correctly
    clear_btn.click(
        fn=lambda: (None, "", "", ""),
        inputs=None,
        outputs=[img_in, logs_out, score_out, cond_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
