import gradio as gr
import numpy as np

# --- ENGINE ---
class ClinicalDetective:
    def analyze_image(self, image):
        if image is None:
            return "⚠️ No image detected.", "0%", "N/A"
        
        # Simulated diagnostic logic
        diagnosis = "Melanocytic Nevus"
        confidence = "92.45%"
        
        report = (
            f"🔬 ANALYSIS REPORT\n"
            f"---------------------------\n"
            f"Diagnosis: {diagnosis}\n"
            f"Confidence: {confidence}"
        )
        return report, confidence, diagnosis

detective = ClinicalDetective()

# --- BLOCKS INTERFACE ---
# By setting show_api=False here, we prevent the generator from 
# trying to parse the component schemas
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

    # Event binding
    analyze_btn.click(
        fn=detective.analyze_image, 
        inputs=img_in, 
        outputs=[logs_out, score_out, cond_out]
    )
    
    clear_btn.click(
        fn=lambda: (None, "", "", ""),
        inputs=None,
        outputs=[img_in, logs_out, score_out, cond_out]
    )

# --- At the bottom of app.py ---
if __name__ == "__main__":
    # Launch with explicit API settings to prevent schema errors
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        show_api=False
    )
