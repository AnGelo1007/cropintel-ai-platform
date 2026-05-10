from pathlib import Path
from datetime import datetime
import subprocess
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for

from services.inference import analyze_image, get_health_status

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")


def _parse_crop_age(raw_value):
    if raw_value is None:
        return None

    cleaned = str(raw_value).strip()
    if cleaned == "":
        return None

    try:
        value = int(cleaned)
        if value < 0:
            return None
        return value
    except (TypeError, ValueError):
        return None


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def home():
    return redirect(url_for("auth_page"))


@app.route("/auth")
def auth_page():
    return render_template("auth.html")


@app.route("/dashboard")
def dashboard():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify(get_health_status())


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file uploaded."}), 400

    image_file = request.files["image"]

    if not image_file or image_file.filename == "":
        return jsonify({"success": False, "error": "Empty image file."}), 400

    filename = image_file.filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = f"{timestamp}_{filename}"
    save_path = UPLOAD_DIR / safe_name

    planting_date = (request.form.get("planting_date") or "").strip() or None
    crop_age_days = _parse_crop_age(request.form.get("crop_age_days"))
    analysis_date = datetime.now().strftime("%Y-%m-%d")

    try:
        image_file.save(save_path)

        result = analyze_image(
            str(save_path),
            planting_date=planting_date,
            crop_age_days=crop_age_days,
            analysis_date=analysis_date,
        )

        if not isinstance(result, dict):
            return jsonify({"success": False, "error": "Invalid analysis response."}), 500

        if not result.get("success", False):
            return jsonify(result), 500

        now = datetime.now()
        result["original_filename"] = filename
        result["saved_filename"] = safe_name
        result["preview_url"] = f"/uploads/{safe_name}"
        result["captured_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
        result["captured_at_ts"] = int(now.timestamp() * 1000)
        result["analysis_date"] = analysis_date
        result["planting_date"] = result.get("planting_date") or planting_date
        result["crop_age_days"] = result.get("crop_age_days", crop_age_days)

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/exit-kiosk", methods=["POST"])
def api_exit_kiosk():
    try:
        subprocess.Popen(["pkill", "chromium-browser"])
        subprocess.Popen(["pkill", "chromium"])
        return jsonify({"success": True, "message": "Kiosk exit command sent."})
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to exit kiosk: {str(e)}"}), 500


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)