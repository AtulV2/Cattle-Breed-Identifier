import os
import json
import cv2
import numpy as np

# Ensure headless and CPU compatibility across diverse deployment environments (e.g. Streamlit Cloud)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from ultralytics import YOLO

# Configuration constants
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Breeds JSON path
BREEDS_FILE = os.path.join(DATA_DIR, 'breeds.json')
if not os.path.exists(BREEDS_FILE):
    alt_breeds = os.path.join(PROJECT_ROOT, 'breeds.json')
    if os.path.exists(alt_breeds):
        BREEDS_FILE = alt_breeds

# Dataset directory
DATASET_DIR = os.path.join(DATA_DIR, 'cattle')
if not os.path.exists(DATASET_DIR):
    alt_cattle = os.path.join(PROJECT_ROOT, 'cattle')
    if os.path.exists(alt_cattle):
        DATASET_DIR = alt_cattle

# Canonical paths inside models/ folder
PRIMARY_WEIGHTS = os.path.join(MODELS_DIR, 'cattle_mobilenetv2.weights.h5')
FALLBACK_WEIGHTS = os.path.join(MODELS_DIR, 'livestock_mobilenetv2.weights.h5')
WEIGHTS_PATH = PRIMARY_WEIGHTS if os.path.exists(PRIMARY_WEIGHTS) else FALLBACK_WEIGHTS
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, 'yolov8s.pt')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_target_breeds(dataset_dir=DATASET_DIR, breeds_file=BREEDS_FILE):
    """
    Get sorted list of 50 cattle breed class names.
    Prefers canonical breeds.json for deployment stability; falls back to folder scan.
    """
    if os.path.exists(breeds_file):
        try:
            with open(breeds_file, 'r', encoding='utf-8') as f:
                breeds = json.load(f)
                if isinstance(breeds, list) and len(breeds) > 0:
                    return breeds
        except Exception:
            pass

    if os.path.exists(dataset_dir):
        return sorted([
            d for d in os.listdir(dataset_dir) 
            if os.path.isdir(os.path.join(dataset_dir, d))
        ])
    return []

class CattleDetector:
    """
    Step 1: Detects and isolates cattle using YOLOv8.
    COCO Class ID 19 is 'cow'.
    """
    def __init__(self, model_name=None):
        if model_name is None:
            model_name = YOLO_MODEL_PATH
        if not os.path.exists(model_name):
            alt_root = os.path.join(PROJECT_ROOT, "yolov8s.pt")
            if os.path.exists(alt_root):
                model_name = alt_root
        self.model = YOLO(model_name)
        self.target_class = 19  # COCO ID 19 is 'cow'

    def detect_and_crop(self, image):
        """
        Returns cropped cow RGB image if detected, otherwise None.
        """
        crop, _, _ = self.detect_full(image)
        return crop

    def detect_full(self, image):
        """
        Runs YOLO detection on an RGB image.
        Returns:
            cropped_image: RGB cropped patch with 10% padding (or None)
            bbox: dict with coordinates and conf (or None)
            annotated_image: RGB image with drawn detection bounding box
        """
        annotated_image = image.copy()
        results = self.model(image, conf=0.20, verbose=False, device='cpu')

        best_crop = None
        best_box_info = None
        max_area = 0

        h, w, _ = image.shape

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == self.target_class:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Calculate area to choose the primary/largest cow if multiple detected
                    area = (x2 - x1) * (y2 - y1)

                    # Draw bounding box on annotated image
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 200, 80), 3)
                    label = f"Cattle: {conf * 100:.1f}%"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(
                        annotated_image, 
                        (x1, max(0, y1 - th - 10)), 
                        (x1 + tw + 10, y1), 
                        (0, 200, 80), 
                        -1
                    )
                    cv2.putText(
                        annotated_image, 
                        label, 
                        (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 255), 
                        2
                    )

                    if area > max_area:
                        max_area = area
                        pad_w = int((x2 - x1) * 0.10)
                        pad_h = int((y2 - y1) * 0.10)

                        x1_pad = max(0, x1 - pad_w)
                        y1_pad = max(0, y1 - pad_h)
                        x2_pad = min(w, x2 + pad_w)
                        y2_pad = min(h, y2 + pad_h)

                        best_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
                        best_box_info = {
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "confidence": round(conf * 100, 2)
                        }

        return best_crop, best_box_info, annotated_image

class ImageHandler:
    """Manages I/O operations for cattle images."""
    @staticmethod
    def load_image(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image not found at path: {filepath}")
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Failed to load image: {filepath}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

class DataPreprocessor:
    """Handles the conversion of raw images into model format."""
    def __init__(self, target_size=IMG_SIZE):
        self.target_size = target_size

    def preprocess(self, image):
        resized_image = cv2.resize(image, self.target_size)
        normalized_image = resized_image.astype(np.float32) / 255.0
        return np.expand_dims(normalized_image, axis=0)

class BreedClassifier:
    """Step 2: Classifies the specific cattle breed using MobileNetV2."""
    def __init__(self, num_classes=50, model_weights_path=None):
        self.num_classes = num_classes
        self.model = self._build_model()
        if model_weights_path is None:
            model_weights_path = WEIGHTS_PATH
        if not os.path.exists(model_weights_path):
            for candidate in [
                os.path.join(MODELS_DIR, "cattle_mobilenetv2.weights.h5"),
                os.path.join(MODELS_DIR, "livestock_mobilenetv2.weights.h5"),
                os.path.join(PROJECT_ROOT, "cattle_mobilenetv2.weights.h5"),
                os.path.join(PROJECT_ROOT, "livestock_mobilenetv2.weights.h5")
            ]:
                if os.path.exists(candidate):
                    model_weights_path = candidate
                    break
        if model_weights_path and os.path.exists(model_weights_path):
            self.load_model(model_weights_path)

    def _build_model(self):
        """Constructs and compiles the MobileNetV2 architecture."""
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
        base_model.trainable = False
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_model(self, weights_path):
        if not os.path.exists(weights_path):
            alt_root = os.path.join(PROJECT_ROOT, os.path.basename(weights_path))
            if os.path.exists(alt_root):
                weights_path = alt_root
        self.model.load_weights(weights_path)

    def train_model(self, train_ds, val_ds, epochs=10, save_path=None):
        """Fits the MobileNetV2 model and saves the trained weights into models/."""
        if save_path is None:
            save_path = WEIGHTS_PATH
        # Ensure parent directory (models/) exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )
        self.model.save_weights(save_path)
        print(f"Weights successfully saved to: {save_path}")
        return history

    def predict(self, processed_image, class_names):
        """Performs breed classification on the cropped image, returning top result."""
        predictions = self.model.predict(processed_image, verbose=0)
        idx = int(np.argmax(predictions[0]))
        conf = float(np.max(predictions[0]))
        breed_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        return breed_name, round(conf * 100, 2)

    def predict_top_k(self, processed_image, class_names, k=3):
        """Returns top-k predictions with confidence percentages."""
        predictions = self.model.predict(processed_image, verbose=0)[0]
        top_indices = np.argsort(predictions)[::-1][:k]
        results = []
        for idx in top_indices:
            breed_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
            conf = float(predictions[idx]) * 100.0
            results.append({
                "breed": breed_name,
                "confidence": round(conf, 2)
            })
        return results
