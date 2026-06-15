import os
import torch
import requests
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModel

# 1. Pipeline Initialization
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/medsiglip-448"
    hf_token = os.getenv("HF_TOKEN")
    
    model = AutoModel.from_pretrained(model_id, token=hf_token).to(device)
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    return model, processor, device

model, processor, device = load_pipeline()

# 2. Core Inference Engine logic for UI
def predict_lesion(image_url, query_1, query_2, query_3):
    try:
        # Stream image asset dynamically
        img = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        queries = [q.strip() for q in [query_1, query_2, query_3] if q.strip()]
        
        # Build tensor matrix
        inputs = processor(
            text=queries, images=[img], padding="max_length", max_length=64, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Apply softmax optimization layer across predictions
            probs = outputs.logits_per_image.softmax(dim=-1)[0].cpu().tolist()
            
        # Return formatted dictionary mapping labels to confidences
        return {queries[idx]: probs[idx] for idx in range(len(queries))}
    except Exception as e:
        return {"Error processing inputs": 1.0}

# 3. Streamlined UI Layout Strategy
with gr.Blocks(title="MedSigLIP Zero-Shot Classifier") as demo:
    gr.Markdown("# Zero-Shot Dermatological MedSigLIP Architecture")
    gr.Markdown("### Evaluating Embedding Alignment & Data Drift in Multimodal Healthcare AI")
    
    with gr.Row():
        with gr.Column():
            url_input = gr.Textbox(
                label="SCIN Target Image URL", 
                value="https://storage.googleapis.com/dx-scin-public-data/dataset/images/3445096909671059178.png"
            )
            q1 = gr.Textbox(label="Diagnostic Prompt 1", value="A skin lesion displaying malignant melanoma characteristics")
            q2 = gr.Textbox(label="Diagnostic Prompt 2", value="A benign intradermal melanocytic nevus")
            q3 = gr.Textbox(label="Diagnostic Prompt 3", value="A normal healthy skin sample")
            submit_btn = gr.Button("Compute Cross-Modal Alignment", variant="primary")
            
        with gr.Column():
            output_labels = gr.Label(label="Normalized Similarity Distribution Profile")

    submit_btn.click(fn=predict_lesion, inputs=[url_input, q1, q2, q3], outputs=output_labels)

if __name__ == "__main__":
    # Launching with share=True creates a public live link right from Google Colab
    demo.launch(share=True)
