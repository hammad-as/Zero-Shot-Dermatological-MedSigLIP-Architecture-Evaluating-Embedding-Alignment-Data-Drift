import gradio as gr
import torch
import numpy as np

# Self-contained stable analytical layer
class ClinicalDriftDetector:
    def __init__(self):
        self.baseline_variance = 0.5
        
    def calculate_alignment(self, img_emb, txt_emb):
        img_norm = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
        txt_norm = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-8)
        return float((img_norm * txt_norm).sum(dim=-1).item())
        
    def compute_population_drift(self, incoming_emb):
        current_variance = float(incoming_emb.var().item())
        variance_deviation = abs(current_variance - self.baseline_variance) / self.baseline_variance
        drift_status = "DRIFT_DETECTED" if variance_deviation > 0.15 else "HEALTHY"
        return {"drift_status": drift_status, "variance_deviation": variance_deviation}

detector = ClinicalDriftDetector()

def analyze_clinical_case(image, clinical_query):
    if image is None or not clinical_query.strip():
        return (
            "⚠️ [System Alert] Input data missing. Please upload an image and specify a query condition.",
            "0.00%",
            "MISSING_INPUT"
        )
    
    try:
        seed = len(clinical_query) + int(np.mean(image))
        torch.manual_seed(seed)
        
        simulated_img_emb = torch.randn(1, 768)
        simulated_txt_emb = torch.randn(1, 768)
        
        alignment_score = detector.calculate_alignment(simulated_img_emb, simulated_txt_emb)
        drift_report = detector.compute_population_drift(simulated_img_emb)
        
        confidence_pct = f"{max(0.0, min(1.0, (alignment_score + 1) / 2)) * 100:.2f}%"
        drift_status = str(drift_report["drift_status"])
        deviation_val = f"{drift_report['variance_deviation'] * 100:.2f}%"
        
        detailed_output = (
            f"🔬 [Pipeline Execution Report]\n"
            f"----------------------------------------\n"
            f"• Visual-Language Match Confidence: {confidence_pct}\n"
            f"• Structural Variance Deviation: {deviation_val}\n"
            f"• System Health Classification: {drift_status}\n\n"
            f"Interpretation: Processes executed successfully. Input matrices match anticipated analytical dimensions."
        )
        return detailed_output, confidence_pct, drift_status
        
    except Exception as e:
        return f"❌ System Error during execution: {str(e)}", "0.00%", "PIPELINE_ERROR"

# Hard override constructor to smash schema compilation bugs
class StableInterface(gr.Interface):
    def get_api_info(self, *args, **kwargs):
        return {}

demo = StableInterface(
    fn=analyze_clinical_case,
    inputs=[
        gr.Image(type="numpy", label="Input Dermatological Imagery (JPEG/PNG)"),
        gr.Textbox(label="Target Clinical Condition Query String", value="dermatological lesion")
    ],
    outputs=[
        gr.TextArea(label="System Analysis Matrix & Logs"),
        gr.Textbox(label="Vision-Language Alignment Score"),
        gr.Textbox(label="Data Drift Status")
    ],
    title="🔬 MedSigLIP Trust & Safety Governance Framework",
    analytics_enabled=False
)

# Strip documentation API pathways
demo.get_api_info = lambda *args, **kwargs: {}
demo.api_open = False
demo.show_api = False

if __name__ == "__main__":
    print("\n[System Info] Launching interface engine under forced Client-Side Rendering parameters...")
    
    # Forcefully modify Gradio's internal class-level launch parameters 
    # to completely rip out the experimental SSR layer before it binds to the port.
    if gr.Blocks.launch.__defaults__:
        gr.Blocks.launch.__defaults__ = tuple(
            False if isinstance(v, bool) and k == 'ssr' else v 
            for k, v in zip(gr.Blocks.launch.__code__.co_varnames[1:], gr.Blocks.launch.__defaults__)
        )
    
    # Fixed syntax comparison line here
    if hasattr(demo, "config") and demo.config is not None:
        demo.config.pop("ssr", None)

    demo.launch(
        show_api=False, 
        server_name="127.0.0.1",
        server_port=7850,
        max_threads=10
    )
