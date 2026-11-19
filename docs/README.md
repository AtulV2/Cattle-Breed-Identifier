# 📚 Livestock Breed Identification — Documentation Hub

Welcome to the technical documentation for the **Livestock Breed Identification System**. This directory contains comprehensive technical guides, architecture specifications, deployment instructions, and dataset references for developers, researchers, and field teams.

---

## 🧭 Documentation Navigation

| Document | Description |
| :--- | :--- |
| **[Architecture & System Design](ARCHITECTURE.md)** | Two-stage Computer Vision pipeline (YOLOv8s + MobileNetV2), flowcharts, design decisions, and Streamlit state management. |
| **[Getting Started & Deployment Guide](GETTING_STARTED.md)** | Step-by-step setup for local execution, CLI scripts (`test_inference.py`, `train_classifier.py`), and Streamlit Cloud deployment. |
| **[Dataset & 50 Breed Catalog](DATASET_AND_BREEDS.md)** | Detailed information on the 50 indigenous and exotic cattle breeds, regional origins, agricultural purpose, and dataset structure. |
| **[Developer API Reference](API_REFERENCE.md)** | Python class and function reference (`src/ml_core.py`), code snippets, programmatic inference examples, and CSV data schemas. |

---

## 🎯 Project Overview & Purpose

### The Agricultural Problem
In agricultural economics and national livestock censuses, accurately recording cattle breeds is critical for:
- **Genetic Preservation:** Protecting indigenous native Indian zebu breeds (*Bos indicus*) adapted to tropical climates and heat stress.
- **Selective Breeding & Artificial Insemination:** Ensuring verified pedigree records for dairy yield optimization.
- **Government Subsidies & Insurance:** Accurate verification for livestock welfare, insurance claims, and census records.

### The Automated Solution
Traditional manual identification requires senior veterinary or zoological expertise, leading to transcription errors and subjectivity. Field photographs frequently suffer from:
- Cluttered backgrounds (barns, pastures, human handlers, fences).
- Non-target animals (goats, dogs, buffaloes).
- Inconsistent camera angles and lighting.

This system solves these issues through a **decoupled two-stage AI architecture**:
1. **Localization Gatekeeper:** An Ultralytics **YOLOv8s** object detector scans the raw input, detects strictly cattle (COCO Class 19), and crops the animal with 10% bounding box padding. Non-cattle images are safely flagged and rejected, eliminating false positives.
2. **Fine-Grained Classification:** A transfer-learned **MobileNetV2** deep neural network analyzes the isolated animal patch across **50 distinct breeds**, producing ranked probabilities and confidence metrics.
3. **Census Logging:** Inference events are compiled into a live session log exportable as a standardized CSV for field audits.

---

## 🏗️ Repository Architecture Summary

```text
Livestock_Breed_ID/
├── .streamlit/
│   └── config.toml             # Streamlit UI theme, emerald palette & server upload limits
├── data/
│   ├── breeds.json             # Canonical list of 50 breeds mapped to model weights
│   ├── sample_images/          # Curated test images (Gir, Sahiwal, Amritmahal)
│   └── cattle/                 # 50-breed sample dataset
├── docs/                       # Comprehensive documentation
│   ├── README.md               # Documentation hub (this file)
│   ├── ARCHITECTURE.md         # Two-stage pipeline and system design
│   ├── GETTING_STARTED.md      # Local setup, CLI usage, and cloud deployment
│   ├── DATASET_AND_BREEDS.md   # 50 breed directory, traits, and dataset details
│   └── API_REFERENCE.md        # Python API documentation and code examples
├── models/
│   ├── yolov8s.pt              # Pre-trained YOLOv8s cattle detection weights (22.6 MB)
│   └── livestock_mobilenetv2.weights.h5 # Pre-trained MobileNetV2 50-breed weights (17.6 MB)
├── scripts/
│   ├── test_inference.py       # Standalone CLI one-shot test script
│   └── train_classifier.py     # Local classifier fine-tuning script
├── src/
│   ├── __init__.py             # Python package marker
│   └── ml_core.py              # Centralized ML pipeline (detector, classifier, preprocessor)
├── app.py                      # Main Streamlit Cloud web application
├── requirements.txt            # Production Python dependencies
├── packages.txt                # Linux OS dependencies for headless OpenCV
├── .gitignore                  # Git ignore rules protecting model weights
└── README.md                   # Project landing page & quickstart
```
