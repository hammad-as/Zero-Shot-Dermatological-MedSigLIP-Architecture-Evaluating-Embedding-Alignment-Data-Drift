# Zero-Shot Dermatological MedSigLIP Architecture: Evaluating Embedding Alignment & Data Drift

## Project Overview & Business Case
This repository implements an end-to-end evaluation pipeline utilizing Google Health’s 800-million-parameter multimodal foundation model (**MedSigLIP-448**). The system benchmarks open-vocabulary diagnostic cross-alignment by mapping raw, unaligned dermatological imagery directly to natural language clinical targets via a shared vision-language vector space. 

By eliminating downstream task-specific fine-tuning bottlenecks, this architecture showcases the utility of multimodal foundation models while aggressively stress-testing the framework against production-level vulnerabilities: **spurious embedding correlations** and **distributional dataset shift**.

## Deployment & Local Replication

This architecture supports two deployment methods: an automated, interactive web UI via Gradio, or a lightweight script-based CLI execution.

### Option A: Interactive Web UI Deployment (Recommended)
This method launches a live browser interface to dynamically test image URLs and custom text queries.

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt

## System Architecture & Data Pipeline
The execution pipeline is constructed natively in PyTorch utilizing a parallel dual-tower transformer encoder configuration:

* **Vision Encoder Layer:** 400M parameter Vision Transformer (ViT) that resizes and processes unaligned multi-resolution diagnostic imagery onto a fixed 448x448 pixel matrix canvas.
* **Text Encoder Layer:** 400M parameter clinical natural language processing tokenization layer designed to embed technical medical string targets.
* **Mathematical Optimization:** Implements a custom linear inference post-processing step executing a PyTorch `softmax` operation across the dot-product similarity matrix. This transforms raw, unnormalized model logits into bounded, human-interpretable clinical confidence intervals (0.0% to 100.0%).

## Benchmarking & Empirical Readout

# Production Hardware Execution Logs
===================================================
             MEDSIGLIP INFERENCE READOUT LOGS
===================================================
[INFO] Initializing Google Foundation Model 'google/medsiglip-448' on CUDA...
[INFO] Fetching target medical imagery from public SCIN archives...
    -> Image 1 loaded. Native Dimensions: 445x153
    -> Image 2 loaded. Native Dimensions: 576x656
[INFO] Passing tensor records to frozen encoder towers...

[+] Zero-Shot Similarity Distribution for Lesion Sample 1:
    - "A skin lesion displaying malignant melanoma..." -> 48.93% [MATCH]
    - "A benign intradermal melanocytic nevus..."        -> 17.34%
    - "A normal healthy skin sample..."                  -> 33.73%
  
[+] Zero-Shot Similarity Distribution for Lesion Sample 2:
    - "A skin lesion displaying malignant melanoma..." -> 16.54%
    - "A benign intradermal melanocytic nevus..."        -> 10.85%
    - "A normal healthy skin sample..."                  -> 72.61% [MATCH]
*============================================================

**Advanced Data Analysis**
Statistical Boundary Ambiguity (Sample 1): While the pipeline correctly identifies the primary pathological concern (Melanoma at 48.93%), the similarity weights exhibit a split distribution, allocating 33.73% confidence to a completely normal control. This highlights the inherent boundary ambiguity of out-of-the-box foundation models when processing complex clinical phenotypes.

Sharp Class Separation (Sample 2): The model showcases excellent negative prediction capability, cleanly isolating the healthy skin sample with high statistical confidence (72.61%).

**MLOps Production Safety & Risk Mitigation Case Studies**

1. The Fallacy of Spurious Embedding Correlations
A high zero-shot confidence interval (e.g., 72.61%) is purely a mathematical indicator of vector coordinate proximity within a shared multi-dimensional embedding space—it is not a verified pathological diagnosis. Vision-language models are highly susceptible to exploiting shortcut features. The vision encoder can artificially inflate alignment scores by latching onto clinically irrelevant pixel artifacts, such as lighting shadows, technician ruler/measurement markings, camera noise, or skin-tone variations, rather than actual anatomical cellular structure.

2. Engineering Against Distributional Dataset Drift
As proven by the raw asset payloads (Image 1 at 445x153 vs. Image 2 at 576x656), real-world data arrives unstandardized. While spatial downsampling satisfies pipeline syntax, migrating this model across distinct healthcare facilities introduces severe Data Drift. Variations in camera hardware vendors (e.g., GE vs. Siemens), lens sensor polarization, and institutional exposure baselines fundamentally alter raw pixel distributions. Without active local validation matrices and continuous performance tracking, distributional shift presents an immediate clinical failure risk.

**Deployment & Local Replication**

This architecture supports two deployment methods: an automated, interactive web UI via Gradio, or a permanent cloud hosting strategy via Hugging Face Spaces.

Option A: Interactive Web UI Deployment
This method launches a live browser interface to dynamically test image URLs and custom text queries.

Install Requirements:

Bash
pip install -r requirements.txt
Inject Credentials & Launch:
Configure your environment token and run the app script:

Bash
export HF_TOKEN="your_huggingface_token_here"
python src/app.py


