# Automated Traffic Violation Detection System 🚦🚗

An end-to-end computer vision and API pipeline designed to automatically detect traffic rule violations from video feeds, extract vehicle license plates via OCR, and issue traffic tickets by sending violation metadata to an external e-challan backend API.

This project represents the Machine Learning inference engine for a complete Automated License Plate Recognition (ALPR) and Traffic Enforcement system.

---

## 🌟 Key Features

1. **Real-time Vehicle & Person Detection:** 
   Utilizes a custom-trained **YOLOv8** model to detect highly specific classes, including `WITHOUT_HELMET`, `USING_MOBILE`, and `MORE_THAN_TWO_PERSONS` (Triple Riding). Includes a `DirectViolationDetector` mapping that integrates natively with custom classes.
2. **Robust Object Tracking:** 
   Integrates **DeepSORT** and **IoU tracking** to assign unique IDs to vehicles and track their trajectories across frames. Includes automatic stale-track cleanup to prevent memory leaks during long video processing.
3. **Spatial Violation Engine:** 
   Uses advanced geometric math (finite line intersection algorithms) to detect dynamic violations, such as **Running a Red Light**, by analyzing the historical trajectory of a vehicle against a dynamically detected stop-line.
4. **License Plate Recognition (OCR) with Fallbacks:** 
   Integrates **EasyOCR** to automatically crop violating vehicles, focus on the license plate region, and extract the alphanumeric plate text using text-cleaning algorithms. Includes a smart caching layer and a fallback mechanism to report the plate as `UNKNOWN` if it cannot be read (allowing a human officer to manually review the evidence image).
5. **Backend API Integration:** 
   Fully decoupled from direct database manipulation. Submits violations via a robust REST API client (`model_api.py`) with support for JSON metadata and multipart evidence image uploads, seamlessly connecting to a Prisma/Node.js backend.

---

## 📁 Project Structure

```text
traffic-Violation-detection/
│
├── main.py                  # Main CLI Inference Pipeline (Video Inference & API Submission)
├── requirements.txt         # Cleaned dependency list
│
├── src/                     # Core Python Modules
│   ├── detection.py         # YOLOv8 wrapper & detection data structures
│   ├── tracking.py          # DeepSORT integration & TrajectoryStore memory management
│   ├── utils.py             # Geometric algorithms (e.g., segment intersection)
│   ├── violations.py        # Violation Engine (Custom Direct classes + Spatial triggers)
│   ├── ocr.py               # PlateRecognizer (EasyOCR license plate extraction)
│   └── model_api.py         # HTTP REST Client for submitting violations and evidence
│
└── notebooks/
    └── ocrtest.ipynb        # Sandbox notebook for testing the OCR on static car photos
```

---

## 🛠️ Tech Stack
* **Computer Vision:** YOLOv8 (Ultralytics), OpenCV
* **Tracking:** DeepSORT (`deep-sort-realtime`), IoU Tracker
* **OCR:** EasyOCR (PyTorch)
* **Integration:** Python `requests` for REST APIs
* **Environment:** Python 3.10+, Virtual Environment (`.venv`)

---

## 🚀 How It Works (The Pipeline)

1. **CLI Invocation:** Run `python main.py` with arguments (e.g., `--video test.mp4 --show`).
2. **Detection:** The YOLO model finds bounding boxes for vehicles and custom violation classes directly in the frame.
3. **Tracking:** Persistent IDs are assigned to vehicles to prevent duplicate tickets, while `TrajectoryStore` records their movement paths.
4. **Rule Evaluation:** 
   - Static custom classes (like `WITHOUT_HELMET`, `MORE_THAN_TWO_PERSONS`) immediately trigger a violation flag via `DirectViolationDetector`.
   - The `ViolationEngine` calculates if a vehicle's trajectory crosses spatial thresholds (e.g., Stop line).
5. **OCR Trigger:** If a vehicle is flagged, `src/ocr.py` crops the vehicle bounding box and extracts the license plate text. If illegible, it falls back to `UNKNOWN`.
6. **Evidence Capture:** A cropped image of the vehicle committing the violation is saved temporarily.
7. **API Submission:** The `ModelViolationClient` constructs the backend payload and pushes the JSON metadata + evidence image to the configured Node.js backend URL (`/api/v1/model/violations`).

---

## ⚙️ Setup & Testing

**1. Environment Setup:**
Create a Python virtual environment and install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**2. Ensure Backend is Running:**
Ensure the E-Challan Node.js Backend is running locally on port `5000` (or configure the `TVD_BACKEND_URL` environment variable).

**3. Run the Inference Pipeline:**
Test the system on a video file with the visual debugger enabled (`--show`):
```bash
python main.py --video test.mp4 --device cpu --weights weights/best.pt --show
```
You will see YOLO bounding boxes drawn in real-time, and you can watch the console as it submits detections to your backend API!