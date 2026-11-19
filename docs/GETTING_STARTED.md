# 🚀 Getting Started & Deployment Guide

This guide walks you through setting up and running the **Livestock Breed Identification System** locally, executing the included command-line scripts, and deploying to **Streamlit Community Cloud (`share.streamlit.io`)**.

---

## 1. Prerequisites

- **Python:** Version `3.10` or `3.11` (recommended for optimal TensorFlow 2.x and PyTorch CPU compatibility).
- **Git:** Installed on your workstation.
- **Operating System:** Windows 10/11, macOS, or Linux (Debian/Ubuntu/CentOS).
- **RAM:** Minimum 4 GB (8 GB+ recommended).
- **Hardware Acceleration:** Zero GPU required; inference runs in under 200ms on modern CPUs.

---

## 2. Local Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/Livestock_Breed_ID.git
cd Livestock_Breed_ID
```

### Step 2: Create a Virtual Environment
It is strongly recommended to isolate dependencies inside a virtual environment:

```bash
# On Windows (PowerShell or Command Prompt):
python -m venv venv
.\venv\Scripts\activate

# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
Install the pinned, production-tested requirements:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> `requirements.txt` specifies `opencv-python-headless` instead of standard `opencv-python`. This avoids graphical UI library dependencies and prevents runtime errors on headless servers and cloud containers.

### Step 4: Run the Streamlit Application
Launch the local web server:
```bash
streamlit run app.py
```

Streamlit will automatically open your default browser to:
```text
Local URL: http://localhost:8501
Network URL: http://<your-local-ip>:8501
```

---

## 3. Command-Line Tools (CLI)

The repository provides two standalone scripts located inside `scripts/`:

### A. One-Shot Testing (`scripts/test_inference.py`)
Quickly evaluate an image without starting the web UI.

```bash
# From project root:
python scripts/test_inference.py

# Or from inside scripts/:
cd scripts
python test_inference.py
```

**Interactive Terminal Walkthrough:**
```text
Initializing models (this may take a moment)...
==================================================
Enter the path to the image: data/sample_images/sample_gir.jpg

Step 1: Detecting cattle in image...
Step 2: Cattle detected! Classifying breed...

--- Output ---
File:           data/sample_images/sample_gir.jpg
Detection:      SUCCESS
Detected Breed: Gir
Confidence:     94.12%
--------------
```

### B. Local Classifier Training (`scripts/train_classifier.py`)
Used to train or fine-tune MobileNetV2 on the dataset located in `data/cattle/`:

```bash
# Run training with default 10 epochs:
python scripts/train_classifier.py

# Specify custom epochs and force overwrite of existing weights:
python scripts/train_classifier.py --epochs 20 --force
```

**CLI Arguments:**
- `--epochs <int>`: Number of training epochs (default: `10`).
- `--batch-size <int>`: Batch size for training (default: `32`).
- `--force`: Force training even if pre-trained weights already exist in `models/`.

*Note: The script automatically creates `models/` if absent and guarantees that trained weights are saved directly to `models/livestock_mobilenetv2.weights.h5`.*

---

## 4. Deploying to Streamlit Community Cloud (`share.streamlit.io`)

Streamlit Community Cloud hosts Python applications directly from your GitHub repository for free.

### Step 1: Initialize Git & Push to GitHub
If your repository is not yet pushed to GitHub, run the following commands:
```bash
git init
git add .
git commit -m "Deploy Livestock Breed ID System"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
git push -u origin main
```

> [!IMPORTANT]
> The pre-trained model weights (`models/yolov8s.pt` at 22.6 MB and `models/livestock_mobilenetv2.weights.h5` at 17.6 MB) are well below GitHub's 100 MB hard limit and are pushed directly without requiring Git LFS.

### Step 2: Configure Streamlit Community Cloud
1. Navigate to **[https://share.streamlit.io/](https://share.streamlit.io/)** and sign in with your GitHub account.
2. Click **"New app"** (or **"Create app"**).
3. Fill in the deployment form:
   - **Repository:** `<YOUR_USERNAME>/<YOUR_REPOSITORY>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Under **Advanced settings**, set Python version to **`3.10`** or **`3.11`**.
5. Click **"Deploy!"**.

### Why Deployment is Seamless:
- **`packages.txt`:** Streamlit Cloud automatically runs `apt-get install -y` on `libgl1` and `libglib2.0-0` before Python dependencies install.
- **`requirements.txt`:** Installs `tensorflow-cpu` (avoiding bulky multi-gigabyte CUDA downloads) and `opencv-python-headless`.
- **`.streamlit/config.toml`:** Applies branding colors, layout width, and sets max file upload size to 50MB.
- **In-Memory Caching:** `@st.cache_resource` ensures models load only once on boot, keeping total container memory well below the 1GB quota.
