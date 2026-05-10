from pathlib import Path
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

saved_model_dir = MODEL_DIR / "crop_classifier_savedmodel"
tflite_model_path = MODEL_DIR / "crop_classifier.tflite"

if not saved_model_dir.exists():
    raise FileNotFoundError(
        f"Missing SavedModel export: {saved_model_dir}\n"
        "Run: python train_crop_classifier.py first"
    )

converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Keep export clean: builtins only
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()
tflite_model_path.write_bytes(tflite_model)

print(f"Saved TFLite model to: {tflite_model_path}")