# Crop Health AI Platform for Raspberry Pi

This starter project gives you a Raspberry Pi-friendly website for:
- uploading or capturing crop images
- classifying the crop with TensorFlow Lite
- estimating crop health
- forecasting yield
- storing results locally in SQLite
- optionally syncing images and records to Supabase free-tier services

## Why this stack

- **Flask** is lightweight and easy to run on Raspberry Pi. citeturn530617search2
- **SQLite** is built into Python and is ideal for local offline storage on the Pi. citeturn530617search3
- **LiteRT / TensorFlow Lite** is the recommended TensorFlow runtime for edge devices such as Raspberry Pi. citeturn900714search0turn900714search2turn900714search14
- **Supabase** has an official Python client and supports database plus file storage on a free plan, which makes it a better fit than Firebase Storage for a strictly low-cost capstone in 2026. citeturn530617search0turn530617search1turn863672search4turn863672search22turn900714search5

## Project structure

```text
crop_ai_platform/
├── app.py
├── requirements.txt
├── .env.example
├── models/
│   ├── crop_classifier.tflite   # add later
│   └── labels.json
├── services/
│   ├── cloud_sync.py
│   ├── db.py
│   └── inference.py
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html
├── uploads/
└── instance/
```

## 1) Install on Raspberry Pi

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libatlas-base-dev
cd ~/crop_ai_platform
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For inference, try `tflite-runtime` first because it is lighter than full TensorFlow on embedded Linux. LiteRT’s Python guide documents this workflow for Linux-based devices. citeturn900714search2

## 2) Run the web app

```bash
cd ~/crop_ai_platform
source .venv/bin/activate
python app.py
```

Open:
- `http://127.0.0.1:5000` on the Pi
- `http://YOUR_PI_IP:5000` from another device on the same network

## 3) Add a real TensorFlow Lite model

Put your trained model here:

```text
models/crop_classifier.tflite
```

The app will automatically switch from demo mode to real model inference when the file exists.

Expected model behavior:
- image input shape such as `224x224x3`
- output: probabilities for labels in `models/labels.json`

## 4) Optional free cloud sync with Supabase

Create a Supabase project and bucket, then copy `.env.example` to `.env` and fill in:

```env
SUPABASE_URL=...
SUPABASE_KEY=...
SUPABASE_BUCKET=crop-images
```

You can initialize the Python client with `create_client()` and upload files through the storage API using `storage.from_(bucket).upload(...)`. citeturn530617search15turn530617search1

Example table to create in Supabase SQL editor:

```sql
create table if not exists crop_records (
  id bigint generated always as identity primary key,
  captured_at timestamp,
  crop_name text,
  crop_confidence float8,
  health_status text,
  health_score float8,
  disease_risk text,
  yield_forecast_kg float8,
  image_path text,
  notes text
);
```

## 5) Camera integration later

You can later replace file upload with Raspberry Pi Camera capture by:
- adding a `/api/capture` route
- invoking `libcamera-still` or Picamera2
- saving directly into `uploads/`
- calling the same analyzer

## 6) Important project note

Right now:
- crop classification can be **real AI** once you add a trained `.tflite` model
- health score is a **computer-vision heuristic** based on image greenness and variation
- yield forecast is a **placeholder heuristic**

For a stronger thesis, the next step is to train:
- one crop classification model
- one disease/health model
- one regression model for yield forecasting

Then export them to `.tflite` and run all three on the Pi.
