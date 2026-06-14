# Zero-Shot Dermatological MedSigLIP Architecture: Evaluating Embedding Alignment & Data Drift

## Project Overview & Business Case
This repository implements an end-to-end evaluation pipeline utilizing Google Health’s 800-million-parameter multimodal foundation model (**MedSigLIP-448**). The system benchmarks open-vocabulary diagnostic cross-alignment by mapping raw, unaligned dermatological imagery directly to natural language clinical targets via a shared vision-language vector space. 

By eliminating downstream task-specific fine-tuning bottlenecks, this architecture showcases the utility of multimodal foundation models while aggressively stress-testing the framework against production-level vulnerabilities: **spurious embedding correlations** and **distributional dataset shift**.

---

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
============================================================
