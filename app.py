"""
app.py
Brain Tumor Detection & Clinical Explainability (XAI) Platform
FastAPI Server backend serving interactive UI and diagnostic Grad-CAM inference endpoints.
"""

import os
import glob
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

# Import Explainable AI engine
from xai_engine import predict_and_explain, get_xai_engine

# Base directories
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DATASET_DIR = BASE_DIR / "Brain_tumor_dataset" / "content"

# Initialize FastAPI application
app = FastAPI(
    title="Brain Tumor Detection & Explainability Platform",
    description="Full-stack clinical decision support web system leveraging YOLO classification and PyTorch Grad-CAM.",
    version="2.0.0",
)

# CORS Middleware (permits cross-origin requests if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """
    Renders the modern, responsive clinical dashboard.
    """
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/predict")
async def predict_tumor(file: UploadFile = File(...)):
    """
    Accepts an MRI scan image (multipart/form-data), performs classification via YOLO,
    generates Grad-CAM visual heatmaps, and returns complete diagnostic metadata.
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file was provided in the upload request.",
        )

    # Validate file extension
    filename = file.filename.lower() if file.filename else ""
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    if not filename.endswith(valid_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Please upload an MRI scan with one of: {', '.join(valid_extensions)}",
        )

    try:
        # Read file contents into bytes
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded image file is empty.",
            )

        # Run model inference and PyTorch Grad-CAM generation
        result_payload = predict_and_explain(image_bytes)
        return JSONResponse(content=result_payload)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error processing MRI scan: {str(e)}",
        )


@app.get("/api/sample/{category}")
async def get_sample_image(category: str):
    """
    Retrieves a sample test MRI scan for immediate quick demonstration.
    Supported categories: glioma, meningioma, pituitary, notumor
    """
    category_map = {
        "glioma": "glioma_tumor",
        "meningioma": "meningioma_tumor",
        "pituitary": "pituitary_tumor",
        "notumor": "no_tumor",
        "no_tumor": "no_tumor",
    }

    folder_name = category_map.get(category.lower())
    if not folder_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown sample category '{category}'. Available: {list(category_map.keys())}",
        )

    search_dirs = [
        DATASET_DIR / "Testing" / folder_name,
        DATASET_DIR / "Training" / folder_name,
    ]

    found_images = []
    for d in search_dirs:
        if d.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                found_images.extend(d.glob(ext))
            if found_images:
                break

    if not found_images:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No sample images found for category '{category}' in dataset directory.",
        )

    sample_path = sorted(found_images)[0]
    return FileResponse(path=str(sample_path), media_type="image/jpeg")


@app.get("/api/health")
async def health_check():
    """
    Platform health and model readiness check.
    """
    engine = get_xai_engine()
    return {
        "status": "healthy",
        "model_loaded": engine.is_custom_trained,
        "classes": engine.classes,
        "device": str(engine.device),
        "target_layer": engine.target_layer.__class__.__name__ if engine.target_layer else None,
    }


if __name__ == "__main__":
    import uvicorn
    # Start web server on port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
