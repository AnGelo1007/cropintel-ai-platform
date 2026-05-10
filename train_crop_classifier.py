from pathlib import Path
import json
import random
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_HEAD = 10
EPOCHS_FINE = 10
SEED = 42
VAL_SPLIT = 0.2

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

CANDIDATE_DATASET_DIRS = [
    BASE_DIR / "dataset_crops",
    BASE_DIR.parent / "dataset_crops",
]

DATASET_DIR = None
for candidate in CANDIDATE_DATASET_DIRS:
    if candidate.exists():
        DATASET_DIR = candidate
        break

if DATASET_DIR is None:
    raise FileNotFoundError(
        "Dataset folder not found.\n\n"
        "Expected one of these:\n"
        f" - {BASE_DIR / 'dataset_crops'}\n"
        f" - {BASE_DIR.parent / 'dataset_crops'}\n"
    )

CROP_NAMES = ["corn", "eggplant", "pepper", "rice", "tomato"]
HEALTH_NAMES = ["healthy", "unhealthy"]


def is_candidate_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def verify_image_file(path: Path) -> bool:
    """
    Strict validation using PIL.
    This catches corrupted or fake image files even if the extension looks correct.
    """
    try:
        with Image.open(path) as img:
            img.verify()

        # Re-open after verify() because verify() closes validation state
        with Image.open(path) as img:
            img.convert("RGB")

        return True
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return False
    except Exception:
        return False


def collect_valid_images(folder: Path):
    valid_files = []
    invalid_files = []

    for path in sorted(folder.iterdir()):
        if not is_candidate_image(path):
            continue

        if verify_image_file(path):
            valid_files.append(path)
        else:
            invalid_files.append(path)

    return valid_files, invalid_files


def scan_dataset_structure(dataset_dir: Path):
    print(f"\nUsing dataset folder: {dataset_dir}\n")
    print("Checking dataset structure...\n")

    combined_class_names = []
    image_counts = {}
    bad_files_report = {}

    for crop in CROP_NAMES:
        crop_dir = dataset_dir / crop
        if not crop_dir.exists():
            raise FileNotFoundError(f"Missing crop folder: {crop_dir}")

        for health in HEALTH_NAMES:
            class_dir = crop_dir / health
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing health folder: {class_dir}")

            class_name = f"{crop}_{health}"
            combined_class_names.append(class_name)

            valid_files, invalid_files = collect_valid_images(class_dir)
            image_counts[class_name] = len(valid_files)

            if invalid_files:
                bad_files_report[class_name] = invalid_files

    if bad_files_report:
        print("Invalid or unreadable image files found and skipped:\n")
        for class_name, bad_files in bad_files_report.items():
            print(f"{class_name}: {len(bad_files)} bad file(s)")
            for bad in bad_files[:20]:
                print(f"  - {bad.name}")
            if len(bad_files) > 20:
                print(f"  ... and {len(bad_files) - 20} more")
            print()

    print("Detected classes and usable image counts:")
    total_images = 0
    for class_name in combined_class_names:
        count = image_counts[class_name]
        total_images += count
        print(f"  {class_name}: {count}")

    print(f"\nTotal usable images found: {total_images}\n")

    empty_classes = [name for name, count in image_counts.items() if count == 0]
    if empty_classes:
        raise ValueError(
            "Some classes have no usable images:\n"
            + "\n".join(f" - {name}" for name in empty_classes)
        )

    return combined_class_names, image_counts


def build_file_list(dataset_dir: Path):
    filepaths = []
    labels = []
    class_names = []

    for crop in CROP_NAMES:
        for health in HEALTH_NAMES:
            class_name = f"{crop}_{health}"
            class_names.append(class_name)

    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    for crop in CROP_NAMES:
        for health in HEALTH_NAMES:
            class_name = f"{crop}_{health}"
            class_dir = dataset_dir / crop / health

            valid_files, _ = collect_valid_images(class_dir)
            for filepath in valid_files:
                filepaths.append(str(filepath))
                labels.append(class_to_index[class_name])

    return filepaths, labels, class_names


def decode_and_resize(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3, try_recover_truncated=True)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def decode_by_extension(path, label):
    path_str = path.numpy().decode("utf-8")
    suffix = Path(path_str).suffix.lower()

    raw = tf.io.read_file(path_str)

    if suffix in {".jpg", ".jpeg"}:
        image = tf.image.decode_jpeg(raw, channels=3, try_recover_truncated=True)
    elif suffix == ".png":
        image = tf.image.decode_png(raw, channels=3)
    elif suffix == ".bmp":
        image = tf.image.decode_bmp(raw)
    else:
        raise ValueError(f"Unsupported extension during decode: {suffix}")

    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image.numpy(), label.numpy()


def tf_decode_wrapper(path, label):
    image, label = tf.py_function(
        func=decode_by_extension,
        inp=[path, label],
        Tout=[tf.float32, tf.int32],
    )
    image.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    label.set_shape(())
    return image, label


def build_tf_dataset(paths, labels, training=False):
    labels = [int(x) for x in labels]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(tf_decode_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=max(len(paths), 100), seed=SEED)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def get_recommendation_guide():
    return {
        "healthy": {
            "summary": "The crop appears healthy.",
            "recommended_actions": [
                "Continue the current good cultivation practices.",
                "Maintain proper watering and fertilization schedule.",
                "Keep monitoring for early signs of pests or disease.",
                "Continue regular field sanitation and inspection.",
                "Harvest at the proper maturity stage."
            ]
        },
        "unhealthy": {
            "summary": "The crop appears unhealthy and may show signs of disease, pests, or nutrient deficiency.",
            "recommended_actions": [
                "Inspect the crop closely for visible lesions, discoloration, rot, or pest damage.",
                "Remove severely damaged or infected crop samples when necessary.",
                "Check for pest presence and apply the appropriate pest management method.",
                "Review fertilizer application and correct possible nutrient deficiency.",
                "Adjust irrigation if the crop is affected by too much or too little water.",
                "Consult an agricultural technician or local expert for further validation if symptoms persist."
            ]
        },
        "crop_specific_notes": {
            "tomato": [
                "Watch for rotting, cracking, black spots, and discoloration.",
                "Avoid excess moisture that may increase disease development."
            ],
            "pepper": [
                "Watch for wrinkling, holes, soft rot, and surface lesions.",
                "Check for insect feeding damage and fungal symptoms."
            ],
            "eggplant": [
                "Watch for holes, scars, dark lesions, and deformities.",
                "Monitor for insect damage and fruit surface discoloration."
            ],
            "corn": [
                "Check kernels or ears for mold, rot, and insect damage.",
                "Keep drying and storage conditions proper after harvest."
            ],
            "rice": [
                "Check grains for discoloration, pest damage, and poor grain filling.",
                "Monitor moisture conditions during storage and post-harvest handling."
            ]
        }
    }


def split_train_val(paths, labels, val_split=VAL_SPLIT):
    total = len(paths)
    indices = list(range(total))
    random.Random(SEED).shuffle(indices)

    val_size = int(total * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_paths = [paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]

    val_paths = [paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    return train_paths, train_labels, val_paths, val_labels


def make_model(num_classes):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.10),
    ], name="data_augmentation")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="image")
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="crop_health_classifier")
    return model, base_model


def build_inference_model(trained_model, base_model):
    infer_inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="image")
    y = tf.keras.applications.mobilenet_v2.preprocess_input(infer_inputs)
    y = base_model(y, training=False)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    y = trained_model.get_layer("predictions")(y)
    infer_model = tf.keras.Model(infer_inputs, y, name="crop_health_classifier_infer")
    return infer_model


def main():
    tf.random.set_seed(SEED)
    random.seed(SEED)

    class_names, image_counts = scan_dataset_structure(DATASET_DIR)
    filepaths, labels, class_names = build_file_list(DATASET_DIR)

    if len(filepaths) < 10:
        raise ValueError(f"Not enough usable images found for training. Total: {len(filepaths)}")

    train_paths, train_labels, val_paths, val_labels = split_train_val(filepaths, labels)

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}\n")

    train_ds = build_tf_dataset(train_paths, train_labels, training=True)
    val_ds = build_tf_dataset(val_paths, val_labels, training=False)

    print("Detected training classes:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
    print()

    model, base_model = make_model(num_classes=len(class_names))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "best_crop_health_classifier.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
    ]

    print("Training classifier head...\n")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nFine-tuning top layers...\n")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation accuracy: {acc:.4f}")
    print(f"Validation loss: {loss:.4f}\n")

    keras_model_path = MODEL_DIR / "crop_health_classifier.keras"
    model.save(keras_model_path)

    infer_model = build_inference_model(model, base_model)
    saved_model_dir = MODEL_DIR / "crop_health_classifier_savedmodel"
    infer_model.export(saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite_model = converter.convert()
    tflite_path = MODEL_DIR / "crop_health_classifier.tflite"
    tflite_path.write_bytes(tflite_model)

    labels_path = MODEL_DIR / "crop_health_labels.txt"
    labels_path.write_text("\n".join(class_names), encoding="utf-8")

    recommendation_guide = get_recommendation_guide()

    meta = {
        "project_task": "crop_health_classification",
        "input_size": [IMG_SIZE[0], IMG_SIZE[1]],
        "class_names": class_names,
        "crop_names": CROP_NAMES,
        "health_names": HEALTH_NAMES,
        "dataset_dir": str(DATASET_DIR),
        "image_counts": image_counts,
        "validation_split": VAL_SPLIT,
        "recommended_actions": recommendation_guide,
        "prediction_output_format": {
            "class_name_example": "tomato_healthy",
            "meaning": {
                "crop_name": "tomato",
                "health_status": "healthy"
            }
        }
    }

    meta_path = MODEL_DIR / "crop_health_model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    actions_path = MODEL_DIR / "recommended_actions.json"
    actions_path.write_text(json.dumps(recommendation_guide, indent=2), encoding="utf-8")

    print(f"Saved Keras model to: {keras_model_path}")
    print(f"Exported inference SavedModel to: {saved_model_dir}")
    print(f"Saved TFLite model to: {tflite_path}")
    print(f"Saved labels to: {labels_path}")
    print(f"Saved metadata to: {meta_path}")
    print(f"Saved recommendation guide to: {actions_path}")

    print("\nTraining complete.")
    print("This model now predicts crop type + health status together.")
    print("Example outputs:")
    print("  tomato_healthy")
    print("  tomato_unhealthy")
    print("  rice_healthy")
    print("  pepper_unhealthy")


if __name__ == "__main__":
    main()