import os
import sys
import argparse
from datetime import datetime

# Resolve absolute project root regardless of CWD (works from root or scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Add src/ to Python path
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Path to cattle dataset
DATASET_DIR = os.path.join(DATA_DIR, "cattle")
if not os.path.exists(DATASET_DIR):
    alt_cattle = os.path.join(PROJECT_ROOT, "cattle")
    if os.path.exists(alt_cattle):
        DATASET_DIR = alt_cattle

# Path where trained weights will be saved (strictly inside models/)
PRIMARY_WEIGHTS = os.path.join(MODELS_DIR, "cattle_mobilenetv2.weights.h5")
FALLBACK_WEIGHTS = os.path.join(MODELS_DIR, "livestock_mobilenetv2.weights.h5")
WEIGHTS_PATH = PRIMARY_WEIGHTS if os.path.exists(PRIMARY_WEIGHTS) else (FALLBACK_WEIGHTS if os.path.exists(FALLBACK_WEIGHTS) else PRIMARY_WEIGHTS)

import tensorflow as tf
from ml_core import (
    ImageHandler, 
    DataPreprocessor, 
    BreedClassifier, 
    get_target_breeds, 
    BATCH_SIZE, 
    IMG_SIZE
)

class CensusRecord:
    """Data Transfer Object (DTO) structuring results for the digital census DB."""
    def __init__(self, breed_name, confidence_level):
        self.breed_name = breed_name
        self.confidence_level = round(float(confidence_level), 2)
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save_to_db(self):
        """Mocks the database injection process."""
        return "Record successfully pushed to Digital Census DB."

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Cattle Breed Classifier")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--force", action="store_true", help="Force retrain and overwrite existing weights in models/")
    args = parser.parse_args()

    print("=" * 60)
    print("      Cattle Breed Identifier - MobileNetV2 Classifier Trainer")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Path: {DATASET_DIR}")
    print(f"Models Path:  {MODELS_DIR}")
    print(f"Weights File: {WEIGHTS_PATH}")
    print("-" * 60)

    # Pre-check dataset directory
    if not os.path.exists(DATASET_DIR):
        print(f"\n[ERROR] Dataset directory not found at: '{DATASET_DIR}'")
        print("Please ensure the 'cattle' dataset folder is located inside 'data/cattle/' or in the project root.\n")
        sys.exit(1)

    # Load dataset
    print("\nLoading dataset from disk...")
    total_files = sum([
        len(files) for r, d, files in os.walk(DATASET_DIR) 
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)
    ])
    print(f"Total valid images found: {total_files}")

    if total_files == 0:
        print("[ERROR] No valid cattle images found in dataset directory.")
        sys.exit(1)

    if total_files > 5:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR, validation_split=0.2, subset="training", seed=123,
            image_size=IMG_SIZE, batch_size=BATCH_SIZE
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR, validation_split=0.2, subset="validation", seed=123,
            image_size=IMG_SIZE, batch_size=BATCH_SIZE
        )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR, seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE
        )
        val_ds = train_ds

    TARGET_BREEDS = train_ds.class_names
    print(f"Detected {len(TARGET_BREEDS)} Breeds: {TARGET_BREEDS[:5]} ...")

    # Normalize datasets
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Initialize Classifier
    classifier = BreedClassifier(num_classes=len(TARGET_BREEDS), model_weights_path=WEIGHTS_PATH)

    # Determine whether to train
    should_train = not os.path.exists(WEIGHTS_PATH) or args.force

    if should_train:
        print(f"\nInitiating Training Phase ({args.epochs} epochs)...")
        # Ensure models/ folder exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        classifier.train_model(train_ds, val_ds, epochs=args.epochs, save_path=WEIGHTS_PATH)
        print(f"\n[SUCCESS] Model weights saved to: {WEIGHTS_PATH}")
    else:
        print(f"\nPre-trained weights found at: {WEIGHTS_PATH}")
        print("Skipping training. To retrain, run: python scripts/train_classifier.py --force")

    # Test inference on a sample image
    print("\n--- Verifying Inference on Sample Image ---")
    image_handler = ImageHandler()
    preprocessor = DataPreprocessor(target_size=IMG_SIZE)

    first_breed_dir = os.path.join(DATASET_DIR, TARGET_BREEDS[0])
    valid_exts = ('.jpg', '.jpeg', '.png')
    sample_images = [f for f in os.listdir(first_breed_dir) if f.lower().endswith(valid_exts)] if os.path.exists(first_breed_dir) else []

    if sample_images:
        test_path = os.path.join(first_breed_dir, sample_images[0])
        try:
            raw_img = image_handler.load_image(test_path)
            tensor = preprocessor.preprocess(raw_img)
            breed, conf = classifier.predict(tensor, TARGET_BREEDS)
            record = CensusRecord(breed_name=breed, confidence_level=conf)
            db_status = record.save_to_db()

            print(f"Sample Tested:  {test_path}")
            print(f"Detected Breed: {record.breed_name}")
            print(f"Confidence:     {record.confidence_level}%")
            print(f"Census Status:  {db_status}")
        except Exception as e:
            print(f"[Inference Error]: {e}")
    else:
        print("No sample images found for verification test.")
