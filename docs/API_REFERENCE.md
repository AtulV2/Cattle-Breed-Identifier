# 💻 Developer API & Module Reference

This document provides a technical API reference for `src/ml_core.py`, detailing class signatures, return structures, and programmatic code examples for developers integrating the pipeline into external applications, mobile backends, or batch scripts.

---

## 1. Module Overview (`src/ml_core.py`)

All core machine learning operations (YOLO object detection, MobileNetV2 classification, tensor preprocessing, and dataset class discovery) are centralized inside `src/ml_core.py`.

### Path Resolution
The module dynamically discovers project directories relative to `src/`:
- `MODELS_DIR` $\rightarrow$ `models/`
- `DATA_DIR` $\rightarrow$ `data/`
- `BREEDS_FILE` $\rightarrow$ `data/breeds.json`
- `WEIGHTS_PATH` $\rightarrow$ `models/cattle_mobilenetv2.weights.h5`
- `YOLO_MODEL_PATH` $\rightarrow$ `models/yolov8s.pt`

---

## 2. Class Reference

### `CattleDetector`
Encapsulates Ultralytics YOLOv8s targeting cattle localization.

```python
from src.ml_core import CattleDetector

detector = CattleDetector(model_name=None)
```

#### Methods:

##### `detect_full(image: np.ndarray) -> tuple`
Runs YOLO detection on an RGB image.
- **Parameters:**
  - `image` (*np.ndarray*): RGB image array of shape `(H, W, 3)` with uint8 values in $[0, 255]$.
- **Returns:**
  - `cropped_image` (*np.ndarray* | *None*): Cropped RGB patch with 10% padding if cattle found, else `None`.
  - `bbox_info` (*dict* | *None*): Dictionary with keys `{"x1", "y1", "x2", "y2", "confidence"}` if found, else `None`.
  - `annotated_image` (*np.ndarray*): Copy of the original RGB image with the green detection bounding box and confidence score drawn.

##### `detect_and_crop(image: np.ndarray) -> np.ndarray | None`
Convenience wrapper returning strictly the padded cropped patch or `None`.

---

### `BreedClassifier`
Encapsulates the fine-tuned MobileNetV2 50-class breed classifier.

```python
from src.ml_core import BreedClassifier

classifier = BreedClassifier(num_classes=50, model_weights_path=None)
```

#### Methods:

##### `predict(processed_image: np.ndarray, class_names: list[str]) -> tuple[str, float]`
Computes the primary (Top-1) predicted breed.
- **Parameters:**
  - `processed_image` (*np.ndarray*): Batch tensor of shape `(1, 224, 224, 3)` normalized to $[0.0, 1.0]$.
  - `class_names` (*list[str]*): List of breed names corresponding to model output indices.
- **Returns:**
  - `breed_name` (*str*): The highest-probability breed name.
  - `confidence` (*float*): Confidence percentage rounded to 2 decimal places (e.g. `94.12`).

##### `predict_top_k(processed_image: np.ndarray, class_names: list[str], k: int = 3) -> list[dict]`
Computes ranked Top-K candidate probabilities.
- **Parameters:**
  - `processed_image` (*np.ndarray*): Batch tensor of shape `(1, 224, 224, 3)`.
  - `class_names` (*list[str]*): List of breed names.
  - `k` (*int*): Number of top candidates to return (default: `3`).
- **Returns:**
  - List of dictionaries: `[{"breed": "Gir", "confidence": 85.34}, {"breed": "Sahiwal", "confidence": 10.12}, ...]`

##### `train_model(train_ds, val_ds, epochs: int = 10, save_path: str = None) -> History`
Fits the classification head on training data and automatically creates the target directory before saving weights.
- **Default `save_path`:** `models/cattle_mobilenetv2.weights.h5`.

##### `load_model(weights_path: str)`
Loads `.weights.h5` weights into the compiled Keras graph with fallback checking.

---

### `DataPreprocessor`
Converts raw RGB numpy crops into model-compatible tensors.

```python
from src.ml_core import DataPreprocessor

preprocessor = DataPreprocessor(target_size=(224, 224))
```

#### Methods:

##### `preprocess(image: np.ndarray) -> np.ndarray`
- Resizes image to $(224, 224)$ via OpenCV bilinear interpolation.
- Converts to float32 and scales pixel intensities by $1 / 255.0$.
- Adds batch dimension, returning a tensor of shape `(1, 224, 224, 3)`.

---

### `ImageHandler`
Utility class for safe file I/O.

##### `load_image(filepath: str) -> np.ndarray`
- Reads image from disk using OpenCV (`cv2.imread`).
- Automatically converts from BGR to RGB color space.
- Raises `FileNotFoundError` or `ValueError` if the file is missing or corrupted.

---

### `get_target_breeds(dataset_dir=None, breeds_file=None) -> list[str]`
Discovers and returns the sorted list of 50 class names.
- **Priority:** Checks `data/breeds.json` first for deterministic class ordering. Falls back to scanning `data/cattle/` if the JSON is missing.

---

## 3. Programmatic Usage Examples

### Example 1: Standalone Two-Stage Inference
```python
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.abspath("src"))

from ml_core import (
    CattleDetector,
    BreedClassifier,
    DataPreprocessor,
    ImageHandler,
    get_target_breeds
)

# 1. Initialize components
breeds = get_target_breeds()
detector = CattleDetector()
classifier = BreedClassifier(num_classes=len(breeds))
preprocessor = DataPreprocessor()
image_handler = ImageHandler()

# 2. Load and inspect image
image_path = "data/sample_images/sample_gir.jpg"
image_rgb = image_handler.load_image(image_path)

# 3. Stage 1: Detect cattle
crop_rgb, bbox_info, _ = detector.detect_full(image_rgb)

if crop_rgb is not None:
    # 4. Stage 2: Classify breed
    tensor = preprocessor.preprocess(crop_rgb)
    top_breed, top_conf = classifier.predict(tensor, breeds)
    top_3 = classifier.predict_top_k(tensor, breeds, k=3)

    print(f"Cattle Confirmed ({bbox_info['confidence']}%)")
    print(f"Primary Breed: {top_breed} ({top_conf}%)")
    print("Candidates:", top_3)
else:
    print("No cattle detected in image. Classification skipped.")
```

---

### Example 2: Batch Directory Processing & CSV Export
```python
import os
import sys
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.abspath("src"))
from ml_core import CattleDetector, BreedClassifier, DataPreprocessor, ImageHandler, get_target_breeds

breeds = get_target_breeds()
detector = CattleDetector()
classifier = BreedClassifier(num_classes=len(breeds))
preprocessor = DataPreprocessor()
handler = ImageHandler()

records = []
input_folder = "my_field_images"

for fname in os.listdir(input_folder):
    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        fpath = os.path.join(input_folder, fname)
        try:
            img = handler.load_image(fpath)
            crop, bbox, _ = detector.detect_full(img)
            
            if crop is not None:
                tensor = preprocessor.preprocess(crop)
                breed, conf = classifier.predict(tensor, breeds)
                status = "Cattle Confirmed"
            else:
                breed, conf = "N/A", "N/A"
                status = "No Cattle Detected"

            records.append({
                "Filename": fname,
                "Status": status,
                "Breed": breed,
                "Confidence": f"{conf}%" if conf != "N/A" else "N/A",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            print(f"Error reading {fname}: {e}")

df = pd.DataFrame(records)
df.to_csv("batch_census_results.csv", index=False)
print(f"Processed {len(df)} images -> batch_census_results.csv")
```

---

## 4. Digital Census Data Schema

The CSV exported by both the Streamlit web application and the batch pipeline adheres to the following audit schema:

| Column Name | Data Type | Description | Example |
|:---|:---|:---|:---|
| `Image` / `Filename` | `string` | Original file name or camera capture ID | `field_capture_20261119.jpg` |
| `Status` | `string` | Detection outcome (`Cattle Confirmed` or `No Cattle Detected`) | `Cattle Confirmed` |
| `Identified Breed` | `string` | Primary predicted breed formatted in title-case | `Gir` |
| `Confidence` | `string` | Classification probability percentage | `94.12%` |
| `Timestamp` | `string` | ISO datetime string recorded upon inference | `2026-11-19 14:35:10` |
