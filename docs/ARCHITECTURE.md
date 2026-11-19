# 🏗️ Architecture & System Design

This document details the software architecture, machine learning pipeline, state management, and algorithmic design choices powering the **Livestock Breed Identification System**.

---

## 1. High-Level System Architecture

The application adopts a **two-stage decoupled pipeline** separating spatial animal detection from fine-grained taxonomic breed classification:

```mermaid
flowchart TD
    subgraph "Stage 0: Input Ingestion"
        A1[Multi-File Uploader] --> A[Raw Image Bytes]
        A2[Live Camera Capture] --> A
        A3[Sample Test Gallery] --> A
        A --> B[PIL Image Conversion: RGB Format]
    end

    subgraph "Stage 1: Cattle Localization & Gatekeeper"
        B --> C[Ultralytics YOLOv8s Engine]
        C -->|Detect COCO Class 19: cow| D{Cattle Found?}
        D -->|No: Confidence < 0.20| E[Flag: No Cattle Detected]
        E --> F[Skip Classification & Alert User]
        D -->|Yes: Select Primary Detection Box| G[Apply 10% Bounding Box Padding]
        G --> H[Crop Isolated Cattle Patch]
    end

    subgraph "Stage 2: Fine-Grained Breed Classification"
        H --> I[DataPreprocessor: Resize 224x224 & Rescale 0-1]
        I --> J[TensorFlow MobileNetV2 Engine]
        J --> K[Softmax Probability Distribution 50 Classes]
        K --> L[Extract Top-1 Prediction & Confidence %]
        K --> M[Extract Top-3 Candidate Probabilities]
    end

    subgraph "Stage 3: Output & Audit Logging"
        L --> N[Render Visual Bounding Box Overlay]
        M --> N
        N --> O[Live UI Display Cards]
        O --> P[Append Session Record to Digital Census Log]
        P --> Q[Download Standardized CSV Report]
    end
```

---

## 2. The Two-Stage Machine Learning Pipeline

### Stage 1: Cattle Localization (Ultralytics YOLOv8s)
- **Model:** `yolov8s.pt` (Small variant, ~22.6 MB).
- **Target COCO Class:** Class ID `19` (`cow`).
- **Detection Sensitivity (`conf=0.20`):** Tuned to detect partially visible animals or cattle standing in distant pastures without triggering false detections on background structures.
- **Multi-Animal Area Selection:** When multiple cattle appear in a single photograph, the detector computes bounding box area `(x2 - x1) * (y2 - y1)` and selects the primary (largest / foreground) animal for classification.
- **Dynamic 10% Padding:**
  $$\text{pad}_w = \lfloor (x_2 - x_1) \times 0.10 \rfloor, \quad \text{pad}_h = \lfloor (y_2 - y_1) \times 0.10 \rfloor$$
  $$x_{1,\text{pad}} = \max(0, x_1 - \text{pad}_w), \quad x_{2,\text{pad}} = \min(W, x_2 + \text{pad}_w)$$
  $$y_{1,\text{pad}} = \max(0, y_1 - \text{pad}_h), \quad y_{2,\text{pad}} = \min(H, y_2 + \text{pad}_h)$$
  *Rationale:* Standard object detector boxes tightly clip ears, dewlaps, humps, or hooves. Hump prominence and dewlap skin folds are essential morphological markers for identifying indigenous Indian zebu breeds (*Bos indicus*). Adding 10% context ensures whole-body anatomical visibility.

### Stage 2: Breed Classification (MobileNetV2 Transfer Learning)
- **Base Network:** MobileNetV2 pre-trained on ImageNet.
  - Efficient depthwise separable convolutions yielding high feature extraction accuracy with minimal parameter count (~3.5M parameters, ~14MB memory footprint).
- **Custom Classification Head:**
  ```text
  Input Tensor: (1, 224, 224, 3)
    └── MobileNetV2 Backbone (Feature Extractor: 7x7x1280)
          └── GlobalAveragePooling2D() -> (1, 1280)
                └── Dense(512, activation='relu') -> (1, 512)
                      └── Dense(50, activation='softmax') -> (1, 50)
  ```
- **Inference Preprocessing:**
  - Resized via OpenCV bilinear interpolation to $(224, 224)$.
  - Normalized from $[0, 255]$ integer pixels to $[0.0, 1.0]$ float32 tensor via elementwise division by $255.0$.

---

## 3. Inference Sequence Diagram

The interaction sequence between user interface events, detection, classification, and session persistence:

```mermaid
sequenceDiagram
    autonumber
    actor Officer as Field Officer / User
    participant App as Streamlit UI (app.py)
    participant State as Session State Cache
    participant Detector as CattleDetector (YOLOv8s)
    participant Classifier as BreedClassifier (MobileNetV2)

    Officer->>App: Upload Photo / Capture Camera Frame
    App->>State: Compute unique key (hash + size)
    alt Result already cached for this image
        State-->>App: Return cached prediction & visualization
    else New Image
        App->>Detector: detect_full(image_rgb)
        Detector-->>App: crop_rgb, bbox_info, annotated_image
        alt Cattle Not Detected
            App->>State: Log "No Cattle Detected"
            App-->>Officer: Display Warning & Skip Classification
        else Cattle Confirmed
            App->>Classifier: predict(tensor, TARGET_BREEDS)
            App->>Classifier: predict_top_k(tensor, TARGET_BREEDS, k=3)
            Classifier-->>App: Top-1 Breed, Confidence %, Top-3 list
            App->>State: Cache result & Append Census Record
            App-->>Officer: Render BBox Overlay, Cropped Cattle & Match Card
        end
    end
```

---

## 4. State Management & Streamlit Cloud Optimization

Deploying to **Streamlit Community Cloud (`share.streamlit.io`)** imposes two technical constraints:
1. **Memory Budget:** Free-tier Linux containers operate under a ~1GB RAM threshold.
2. **Execution Model:** Streamlit re-runs scripts from top to bottom on every user interaction (clicks, text input, tab switches).

### Optimizations Implemented:

#### 1. In-Memory Singleton Caching (`@st.cache_resource`)
```python
@st.cache_resource(show_spinner="Loading models...")
def load_ml_pipeline():
    detector = CattleDetector()
    classifier = BreedClassifier(num_classes=50)
    ...
    return detector, classifier, preprocessor, breeds
```
- Prevents loading multiple duplicate instances of PyTorch and TensorFlow graphs into RAM across sessions.
- Pre-warms weights once during initial container boot.

#### 2. Unique Image Hashing & Caching
To prevent re-running inference on every keystroke or tab switch:
- Each image receives an identification key:
  - File Uploads: `upload_{filename}_{size}`
  - Camera Captures: `cam_{md5_hash(bytes)}`
  - Sample Gallery: `sample_{filename}`
- Results are stored in `st.session_state["processed_cache"]`. On rerun, results render instantaneously without CPU inference overhead.

#### 3. State-Safe Component Reset
- The file uploader and camera input are bound to dynamic version keys:
  - `st.file_uploader(..., key=f"uploader_{st.session_state['uploader_key']}")`
  - `st.camera_input(..., key=f"camera_{st.session_state['camera_key']}")`
- When the user clicks **"🔄 Clear & Reset All"**, the application increments both keys, freeing the uploaded file and browser camera frame from memory while emptying `census_log`.

---

## 5. Deployment & Runtime Specifications

| Dimension | Specification | Details |
| :--- | :--- | :--- |
| **Application Process** | Single Python process | `streamlit run app.py` serving both UI and in-memory inference |
| **Cloud Target** | Streamlit Community Cloud | Native deployment from GitHub via `share.streamlit.io` |
| **Model Storage** | `models/` directory | `yolov8s.pt` (22.6 MB) and `livestock_mobilenetv2.weights.h5` (17.6 MB) |
| **Inference Mode** | CPU-Optimized | Vectorized NumPy / TensorFlow execution (`CUDA_VISIBLE_DEVICES = -1`) |
| **Explainability** | Top-1 + Top-3 distribution | Primary match with confidence badge + candidate probabilities |
| **Audit Logging** | In-Memory Session Log | Interactive tabular view with standardized CSV export |
