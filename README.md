# Brain Tumor Detection & Clinical Explainability (XAI) Platform

An end-to-end full-stack deep learning and Explainable AI (XAI) application for brain tumor classification and clinical decision support. The platform integrates **Ultralytics YOLO Classification** with **PyTorch Grad-CAM**, served through an asynchronous **FastAPI** backend and an interactive, responsive medical-grade dashboard.

---

## 1. System Architecture

```
                    +--------------------------------------------------+
                    |             Medical Web Dashboard                |
                    |   (Tailwind CSS, Lucide Icons, Glassmorphism)   |
                    +-------------------------+------------------------+
                                              |
                             HTTP POST /api/predict (Multipart Form)
                                              |
                                              v
                    +--------------------------------------------------+
                    |             FastAPI Server (app.py)              |
                    +-------------------------+------------------------+
                                              |
                              Inference & Grad-CAM Generation
                                              |
                                              v
                    +--------------------------------------------------+
                    |           XAI Engine (xai_engine.py)             |
                    |                                                  |
                    |  +--------------------+   +-------------------+  |
                    |  |  YOLOv8 Backbone   |-->| PyTorch Grad-CAM  |  |
                    |  |  Feature Extractor |   |   (Last C2f Conv) |  |
                    |  +--------------------+   +-------------------+  |
                    |            |                        |            |
                    |            v                        v            |
                    |     Class Softmax           Saliency Heatmap     |
                    |     Probabilities           (Jet Color Overlay)  |
                    +-------------------------+------------------------+
                                              |
                               JSON Response + Base64 Images
                                              |
                                              v
                    +--------------------------------------------------+
                    |   Side-by-Side Visual Inspection & Diagnosis     |
                    +--------------------------------------------------+
```

### Diagnostic Target Classes
The model classifies brain MRI scans into four clinical categories:
1. **`glioma_tumor`**: Intra-axial glial neoplasms (astrocytoma, oligodendroglioma, glioblastoma).
2. **`meningioma_tumor`**: Extra-axial meningeal tumors originating from arachnoid cells.
3. **`pituitary_tumor`**: Sellar/suprasellar adenomas affecting endocrine function.
4. **`no_tumor`**: Unremarkable brain neuroanatomy without focal pathology.

---

## 2. Directory Structure

```
Mini_Project/
├── Brain_tumor_dataset/
│   └── content/
│       ├── Training/
│       │   ├── glioma_tumor/
│       │   ├── meningioma_tumor/
│       │   ├── no_tumor/
│       │   └── pituitary_tumor/
│       └── Testing/
│           ├── glioma_tumor/
│           ├── meningioma_tumor/
│           ├── no_tumor/
│           └── pituitary_tumor/
├── static/
│   ├── css/
│   │   └── styles.css          # Custom styling, dark/light theme, animations
│   └── js/
│       └── main.js             # Drag-and-drop, API integration, XAI rendering
├── templates/
│   └── index.html              # Responsive medical-grade web dashboard
├── weights/
│   ├── best.pt                 # Saved trained model weights
│   ├── confusion_matrix.png    # Evaluation confusion matrix heatmap
│   └── evaluation_report.json  # Quantitative metrics (Top-1, Precision, Recall)
├── app.py                      # FastAPI application server
├── train.py                    # Model training & evaluation pipeline
├── xai_engine.py               # PyTorch Grad-CAM explainability module
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation and usage guide
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+
- (Optional) CUDA-enabled GPU for accelerated training

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Model Training & Evaluation

The training script `train.py` automates the entire lifecycle:
- Loads pretrained `yolov8m-cls.pt` (or `yolo11m-cls.pt`).
- Maps `Brain_tumor_dataset/content` training and testing splits.
- Trains for 30 epochs with early stopping (`patience=10`), batch size 16, and input resolution 224x224.
- Evaluates on `content/Testing` and computes **Top-1 Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix**.
- Automatically exports weights to `weights/best.pt`.

### Run Training:
```bash
python train.py --epochs 30 --batch 16 --imgsz 224 --patience 10
```

### Run Evaluation Only (using existing checkpoint):
```bash
python train.py --eval-only
```

---

## 5. Running the Web Application

Start the FastAPI application with Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Once started, open your browser and navigate to:
```
http://localhost:8000
```

---

## 6. API Reference

### `GET /`
- Serves the frontend user interface.

### `POST /api/predict`
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (MRI scan image in `.jpg`, `.jpeg`, or `.png` format)
- **Response Format**:
```json
{
  "prediction": "glioma_tumor",
  "confidence": 97.4,
  "probabilities": {
    "glioma_tumor": 0.974,
    "meningioma_tumor": 0.015,
    "no_tumor": 0.006,
    "pituitary_tumor": 0.005
  },
  "original_image": "data:image/jpeg;base64,...",
  "heatmap_image": "data:image/jpeg;base64,..."
}
```

### `GET /api/sample/{category}`
- Returns a sample test image for quick testing (`glioma`, `meningioma`, `pituitary`, `notumor`).

### `GET /api/health`
- Returns system health, device info (CUDA/CPU), and model status.
