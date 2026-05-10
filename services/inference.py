import json
import os
from datetime import datetime, timedelta

import numpy as np
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "crop_health_classifier.tflite")
LABELS_PATH = os.path.join(MODELS_DIR, "crop_health_labels.txt")
META_PATH = os.path.join(MODELS_DIR, "crop_health_model_meta.json")
ACTIONS_PATH = os.path.join(MODELS_DIR, "recommended_actions.json")

Interpreter = None
interpreter = None
input_details = None
output_details = None
class_names = []
model_meta = {}
recommended_actions = {}

model_loaded = False
model_error = None

MIN_CONFIDENCE_THRESHOLD = 0.70


def _resolve_interpreter():
    global Interpreter

    errors = []

    try:
        from tensorflow.lite.python.interpreter import Interpreter as TfInterpreter
        Interpreter = TfInterpreter
        return
    except Exception as e:
        errors.append(f"tensorflow.lite.python.interpreter failed: {e}")

    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        return
    except Exception as e:
        errors.append(f"tf.lite.Interpreter failed: {e}")

    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteRuntimeInterpreter
        Interpreter = TFLiteRuntimeInterpreter
        return
    except Exception as e:
        errors.append(f"tflite_runtime.interpreter failed: {e}")

    raise RuntimeError(" | ".join(errors))


def _load_labels():
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]

    if not labels:
        raise ValueError("crop_health_labels.txt is empty.")

    return labels


def _load_meta():
    if not os.path.exists(META_PATH):
        return {}

    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_actions():
    if not os.path.exists(ACTIONS_PATH):
        return {}

    with open(ACTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _initialize_model():
    global interpreter, input_details, output_details
    global class_names, model_meta, recommended_actions
    global model_loaded, model_error

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        _resolve_interpreter()

        class_names = _load_labels()
        model_meta = _load_meta()
        recommended_actions = _load_actions()

        interpreter_instance = Interpreter(model_path=MODEL_PATH)
        interpreter_instance.allocate_tensors()

        in_details = interpreter_instance.get_input_details()
        out_details = interpreter_instance.get_output_details()

        if not in_details:
            raise RuntimeError("Model input details are empty.")
        if not out_details:
            raise RuntimeError("Model output details are empty.")

        interpreter = interpreter_instance
        input_details = in_details
        output_details = out_details

        model_loaded = True
        model_error = None

    except Exception as e:
        interpreter = None
        input_details = None
        output_details = None
        model_loaded = False
        model_error = str(e)


_initialize_model()


def _softmax(x):
    x = np.array(x, dtype=np.float32)
    x = x - np.max(x)
    exp_x = np.exp(x)
    denom = np.sum(exp_x)
    if denom == 0:
        return np.zeros_like(x)
    return exp_x / denom


def _get_input_shape():
    if not input_details:
        raise RuntimeError("Model input details are unavailable.")

    shape = input_details[0]["shape"]
    if len(shape) != 4:
        raise RuntimeError(f"Unexpected model input shape: {shape}")

    _, height, width, channels = shape
    return int(height), int(width), int(channels)


def _preprocess_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    height, width, channels = _get_input_shape()

    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height))
    image_np = np.array(image)

    if channels == 1:
        image_np = np.mean(image_np, axis=2, keepdims=True)

    image_np = np.expand_dims(image_np, axis=0)

    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.float32:
        image_np = image_np.astype(np.float32)
    elif input_dtype == np.uint8:
        image_np = image_np.astype(np.uint8)
    else:
        image_np = image_np.astype(input_dtype)

    return image_np


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")


def _parse_combined_class_name(class_name: str):
    normalized = _normalize_label(class_name)

    if normalized.endswith("_healthy"):
        crop_name = normalized[:-8]
        health_status = "healthy"
        return crop_name, health_status

    if normalized.endswith("_unhealthy"):
        crop_name = normalized[:-10]
        health_status = "unhealthy"
        return crop_name, health_status

    return normalized, "unknown"


def _crop_profile(crop_name: str):
    crop = _normalize_label(crop_name)

    profiles = {
        "corn": {
            "base_yield": 18.0,
            "total_lifecycle_days": 110,
            "default_days_remaining": 95,
            "stages": [
                {"name": "Seedling", "min_day": 0, "max_day": 14},
                {"name": "Vegetative", "min_day": 15, "max_day": 49},
                {"name": "Tasseling and Silking", "min_day": 50, "max_day": 75},
                {"name": "Grain Filling to Maturity", "min_day": 76, "max_day": 110},
            ],
            "healthy_message": "Corn appears healthy. Continue proper nutrient management, watering, and regular monitoring.",
            "unhealthy_message": "Corn appears unhealthy. Possible presence of disease, pests, or nutrient deficiency.",
        },
        "eggplant": {
            "base_yield": 12.0,
            "total_lifecycle_days": 120,
            "default_days_remaining": 80,
            "stages": [
                {"name": "Seedling", "min_day": 0, "max_day": 20},
                {"name": "Vegetative", "min_day": 21, "max_day": 49},
                {"name": "Flowering", "min_day": 50, "max_day": 74},
                {"name": "Fruiting to Harvest", "min_day": 75, "max_day": 120},
            ],
            "healthy_message": "Eggplant appears healthy. Continue proper crop care and regular monitoring.",
            "unhealthy_message": "Eggplant appears unhealthy. Possible presence of disease, pests, or nutrient deficiency.",
        },
        "pepper": {
            "base_yield": 10.0,
            "total_lifecycle_days": 110,
            "default_days_remaining": 85,
            "stages": [
                {"name": "Seedling", "min_day": 0, "max_day": 20},
                {"name": "Vegetative", "min_day": 21, "max_day": 44},
                {"name": "Flowering", "min_day": 45, "max_day": 69},
                {"name": "Fruiting to Harvest", "min_day": 70, "max_day": 110},
            ],
            "healthy_message": "Pepper appears healthy. Maintain your current growing practices and keep monitoring.",
            "unhealthy_message": "Pepper appears unhealthy. Possible presence of disease, pests, or nutrient deficiency.",
        },
        "rice": {
            "base_yield": 25.0,
            "total_lifecycle_days": 115,
            "default_days_remaining": 110,
            "stages": [
                {"name": "Seedling", "min_day": 0, "max_day": 20},
                {"name": "Tillering", "min_day": 21, "max_day": 45},
                {"name": "Panicle Initiation", "min_day": 46, "max_day": 75},
                {"name": "Grain Filling to Maturity", "min_day": 76, "max_day": 115},
            ],
            "healthy_message": "Rice appears healthy. Continue balanced crop management and routine observation.",
            "unhealthy_message": "Rice appears unhealthy. Possible presence of disease, pests, or nutrient deficiency.",
        },
        "tomato": {
            "base_yield": 14.0,
            "total_lifecycle_days": 100,
            "default_days_remaining": 78,
            "stages": [
                {"name": "Seedling", "min_day": 0, "max_day": 15},
                {"name": "Vegetative", "min_day": 16, "max_day": 35},
                {"name": "Flowering", "min_day": 36, "max_day": 55},
                {"name": "Fruiting to Harvest", "min_day": 56, "max_day": 100},
            ],
            "healthy_message": "Tomato appears healthy. Continue balanced nutrition, watering, and regular observation.",
            "unhealthy_message": "Tomato appears unhealthy. Possible presence of disease, pests, or nutrient deficiency.",
        },
    }

    default_profile = {
        "base_yield": 10.0,
        "total_lifecycle_days": 90,
        "default_days_remaining": 75,
        "stages": [
            {"name": "Early Growth", "min_day": 0, "max_day": 20},
            {"name": "Vegetative", "min_day": 21, "max_day": 45},
            {"name": "Flowering / Reproductive", "min_day": 46, "max_day": 70},
            {"name": "Harvest Window", "min_day": 71, "max_day": 90},
        ],
        "healthy_message": "Crop appears healthy. Continue proper crop care and regular monitoring.",
        "unhealthy_message": "Crop appears unhealthy. Possible presence of disease, pests, or nutrient deficiency.",
    }

    return profiles.get(crop, default_profile)


def _derive_health_score_from_prediction(health_status: str, confidence: float):
    confidence = max(0.0, min(1.0, float(confidence)))

    if health_status == "healthy":
        base = 82
        score = base + round(confidence * 18)
    elif health_status == "unhealthy":
        base = 35
        score = base + round((1.0 - confidence) * 15)
    else:
        score = 50

    score = max(0, min(100, score))
    return score


def _get_action_recommendations(crop_name: str, health_status: str):
    crop_key = _normalize_label(crop_name)
    health_key = _normalize_label(health_status)

    crop_actions = recommended_actions.get(crop_key, {})
    actions = crop_actions.get(health_key, {})

    return {
        "summary": actions.get("summary", ""),
        "recommended_actions": actions.get("recommended_actions", []),
        "crop_specific_notes": actions.get("crop_specific_notes", []),
    }


def _parse_date_string(value):
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    return None


def _resolve_crop_timing(planting_date=None, crop_age_days=None, analysis_date=None):
    analysis_dt = _parse_date_string(analysis_date) if analysis_date else datetime.now()
    planting_dt = _parse_date_string(planting_date)

    normalized_age = None
    if crop_age_days not in (None, ""):
        try:
            normalized_age = max(0, int(float(crop_age_days)))
        except (TypeError, ValueError):
            normalized_age = None

    if planting_dt is not None and normalized_age is None:
        normalized_age = max(0, (analysis_dt.date() - planting_dt.date()).days)

    if normalized_age is not None and planting_dt is None:
        planting_dt = analysis_dt - timedelta(days=normalized_age)

    planting_date_out = planting_dt.strftime("%Y-%m-%d") if planting_dt else None
    analysis_date_out = analysis_dt.strftime("%Y-%m-%d")

    return {
        "analysis_dt": analysis_dt,
        "analysis_date": analysis_date_out,
        "planting_dt": planting_dt,
        "planting_date": planting_date_out,
        "crop_age_days": normalized_age,
    }


def _determine_growth_stage(crop_profile: dict, crop_age_days):
    if crop_age_days is None:
        return "Estimated from image only"

    stages = crop_profile.get("stages", [])
    for stage in stages:
        min_day = int(stage.get("min_day", 0))
        max_day = int(stage.get("max_day", 0))
        if min_day <= crop_age_days <= max_day:
            return stage.get("name", "Unknown Stage")

    if stages and crop_age_days > int(stages[-1].get("max_day", crop_age_days)):
        return stages[-1].get("name", "Maturity")

    return "Unknown Stage"


def _derive_days_remaining(crop_profile: dict, crop_age_days):
    total_lifecycle_days = int(crop_profile.get("total_lifecycle_days", 90))
    default_days_remaining = int(crop_profile.get("default_days_remaining", total_lifecycle_days))

    if crop_age_days is None:
        return max(0, default_days_remaining), total_lifecycle_days, None

    days_remaining = max(0, total_lifecycle_days - int(crop_age_days))
    maturity_progress = min(1.0, max(0.0, float(crop_age_days) / float(total_lifecycle_days)))
    return days_remaining, total_lifecycle_days, maturity_progress


def _derive_yield_forecast(base_yield: float, health_status: str, confidence: float, maturity_progress=None):
    confidence = max(0.0, min(1.0, float(confidence)))

    if health_status == "healthy":
        health_factor = 0.88 + (confidence * 0.12)
    elif health_status == "unhealthy":
        health_factor = 0.45 + ((1.0 - confidence) * 0.15)
    else:
        health_factor = 0.60

    if maturity_progress is None:
        maturity_factor = 0.94
    else:
        maturity_factor = 0.78 + (0.22 * maturity_progress)

    return round(base_yield * health_factor * maturity_factor, 2)


def _derive_yield_estimate_range(base_yield: float, health_status: str, confidence: float, maturity_progress=None):
    center_value = _derive_yield_forecast(
        base_yield=base_yield,
        health_status=health_status,
        confidence=confidence,
        maturity_progress=maturity_progress,
    )

    spread = max(1.0, round(center_value * 0.15, 2))
    min_value = max(0.0, round(center_value - spread, 2))
    max_value = round(center_value + spread, 2)

    if center_value >= 18:
        estimate_level = "High"
    elif center_value >= 10:
        estimate_level = "Moderate"
    else:
        estimate_level = "Low"

    estimate_text = f"{min_value:.2f} - {max_value:.2f} kg"

    if health_status == "healthy":
        estimate_note = (
            "This is an estimated yield range based on crop type, crop health confidence, and growth progress."
        )
    elif health_status == "unhealthy":
        estimate_note = (
            "This is an estimated yield range. Detected crop health issues may reduce the expected harvest."
        )
    else:
        estimate_note = (
            "This is an estimated yield range based on the available crop analysis data."
        )

    return {
        "yield_estimate_range": estimate_text,
        "yield_estimate_min_kg": min_value,
        "yield_estimate_max_kg": max_value,
        "yield_estimate_center_kg": round(center_value, 2),
        "yield_estimate_level": estimate_level,
        "yield_estimate_note": estimate_note,
    }


def _forecast_date_from_analysis(analysis_dt, days_remaining: int):
    target_date = analysis_dt + timedelta(days=int(days_remaining))
    return target_date.strftime("%Y-%m-%d")


def get_health_status():
    input_dtype = None
    output_dtype = None
    input_shape = None

    if input_details:
        input_dtype = str(input_details[0].get("dtype"))
        shape = input_details[0].get("shape")
        input_shape = shape.tolist() if hasattr(shape, "tolist") else shape

    if output_details:
        output_dtype = str(output_details[0].get("dtype"))

    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat() + "Z",
        "cloud_sync_enabled": False,
        "model_loaded": model_loaded,
        "model_error": model_error,
        "model_path": MODEL_PATH,
        "labels_path": LABELS_PATH,
        "meta_path": META_PATH,
        "actions_path": ACTIONS_PATH,
        "labels_loaded": len(class_names),
        "input_shape": input_shape,
        "input_dtype": input_dtype,
        "output_dtype": output_dtype,
        "confidence_threshold_percent": round(MIN_CONFIDENCE_THRESHOLD * 100, 2),
        "project_task": model_meta.get("project_task", "crop_health_classification"),
    }


def analyze_image(image_path: str, planting_date=None, crop_age_days=None, analysis_date=None):
    global model_loaded, model_error

    if not model_loaded or interpreter is None:
        return {
            "success": False,
            "error": "Model is not loaded.",
            "model_loaded": model_loaded,
            "model_error": model_error,
        }

    try:
        input_tensor = _preprocess_image(image_path)

        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()

        raw_output = interpreter.get_tensor(output_details[0]["index"])[0]

        if np.ndim(raw_output) != 1:
            raw_output = np.array(raw_output).flatten()

        if len(raw_output) == 0:
            raise RuntimeError("Model output is empty.")

        probabilities = _softmax(raw_output)

        if len(probabilities) != len(class_names):
            raise RuntimeError(
                f"Output classes ({len(probabilities)}) do not match labels ({len(class_names)})."
            )

        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])
        predicted_class = class_names[best_idx]
        low_confidence = confidence < MIN_CONFIDENCE_THRESHOLD

        top_indices = np.argsort(probabilities)[::-1]
        class_probabilities = [
            {
                "class_name": class_names[int(idx)],
                "confidence": round(float(probabilities[int(idx)]), 4),
                "confidence_percent": round(float(probabilities[int(idx)]) * 100, 2),
            }
            for idx in top_indices
        ]

        crop_name, health_status = _parse_combined_class_name(predicted_class)
        crop_profile = _crop_profile(crop_name)
        timing = _resolve_crop_timing(
            planting_date=planting_date,
            crop_age_days=crop_age_days,
            analysis_date=analysis_date,
        )

        growth_stage = _determine_growth_stage(crop_profile, timing["crop_age_days"])
        days_remaining, total_lifecycle_days, maturity_progress = _derive_days_remaining(
            crop_profile, timing["crop_age_days"]
        )
        forecast_date = _forecast_date_from_analysis(timing["analysis_dt"], days_remaining)
        health_score = _derive_health_score_from_prediction(health_status, confidence)
        yield_estimate = _derive_yield_estimate_range(
            base_yield=float(crop_profile.get("base_yield", 10.0)),
            health_status=health_status,
            confidence=confidence,
            maturity_progress=maturity_progress,
        )

        action_recommendations = _get_action_recommendations(crop_name, health_status)
        recommendations = action_recommendations["recommended_actions"]
        crop_specific_notes = action_recommendations["crop_specific_notes"]

        if health_status == "healthy":
            possible_issue = crop_profile.get(
                "healthy_message",
                "Crop appears healthy. Continue monitoring and proper care."
            )
            disease_risk = "Low"
        elif health_status == "unhealthy":
            possible_issue = crop_profile.get(
                "unhealthy_message",
                "Crop appears unhealthy. Possible presence of disease, pests, or nutrient deficiency."
            )
            disease_risk = "High"
        else:
            possible_issue = "Health status could not be fully determined."
            disease_risk = "Unknown"

        if not recommendations:
            if health_status == "healthy":
                recommendations = [
                    "Continue proper watering, fertilization, and routine monitoring.",
                    "Inspect leaves and stems regularly for early signs of stress or pests.",
                    "Keep the planting area clean and free from weeds."
                ]
            elif health_status == "unhealthy":
                recommendations = [
                    "Inspect the crop for pests, diseases, or nutrient deficiency symptoms.",
                    "Remove severely affected leaves or plants when necessary.",
                    "Consult an agricultural expert if symptoms continue or spread."
                ]
            else:
                recommendations = [
                    "Monitor the crop closely and check it again with another image if needed."
                ]

        confidence_note = None
        if low_confidence:
            confidence_note = (
                f"Low model confidence ({round(confidence * 100, 2)}%). "
                f"Please verify this result manually."
            )

        advisory_note = action_recommendations["summary"] or possible_issue

        return {
            "success": True,
            "image_path": image_path,
            "crop_name": crop_name,
            "health_status": health_status,
            "growth_stage": growth_stage,
            "predicted_class_name": predicted_class,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "health_score": health_score,
            "disease_risk": disease_risk,
            "possible_issue": possible_issue,
            "yield_forecast_kg": yield_estimate["yield_estimate_center_kg"],
            "yield_estimate_range": yield_estimate["yield_estimate_range"],
            "yield_estimate_min_kg": yield_estimate["yield_estimate_min_kg"],
            "yield_estimate_max_kg": yield_estimate["yield_estimate_max_kg"],
            "yield_estimate_center_kg": yield_estimate["yield_estimate_center_kg"],
            "yield_estimate_level": yield_estimate["yield_estimate_level"],
            "yield_estimate_note": yield_estimate["yield_estimate_note"],
            "forecast_date": forecast_date,
            "forecast_days_remaining": days_remaining,
            "estimated_total_lifecycle_days": total_lifecycle_days,
            "planting_date": timing["planting_date"],
            "crop_age_days": timing["crop_age_days"],
            "analysis_date": timing["analysis_date"],
            "recommendations": recommendations,
            "crop_specific_notes": crop_specific_notes,
            "top_predictions": class_probabilities[:5],
            "model_loaded": model_loaded,
            "model_error": model_error,
            "advisory_note": advisory_note,
            "confidence_note": confidence_note,
            "is_confident_prediction": not low_confidence,
            "confidence_threshold": round(MIN_CONFIDENCE_THRESHOLD * 100, 2),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to analyze image: {str(e)}",
            "model_loaded": model_loaded,
            "model_error": model_error,
        }