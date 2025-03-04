from fastapi import APIRouter, UploadFile, Form, Response
import io
from PIL import Image
import logging
import numpy as np
import cv2
import torch
import mediapipe as mp
from backend.model import load_sd_model  # ✅ Centralized model loading

logger = logging.getLogger("AI-Styler")
router = APIRouter()

# ✅ Load both SDXL Base & Refiner from model.py
sdxl_base, sdxl_refiner = load_sd_model()

def detect_face_mask(image):
    """ Detects faces using MediaPipe and generates a precise mask. """
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    image_np = np.array(image)
    results = face_detection.process(image_np)

    mask = np.zeros(image.size[::-1], dtype=np.uint8)  # Black background
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w = image.size[::-1]
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)
            # Expand mask slightly to cover the entire face
            cv2.rectangle(mask, (x, y), (x + width, y + height), 255, -1)

    # Refine mask
    mask = refine_mask(mask)
    return Image.fromarray(mask)

def refine_mask(mask):
    """Refines the mask using morphological operations."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)  # Expand mask
    mask = cv2.erode(mask, kernel, iterations=1)   # Shrink mask
    return mask

@router.post("/stylize-imageStable/")
async def stylize_image(
    file: UploadFile, 
    prompt: str = Form(...), 
    strength_base: float = Form(0.7), 
    guidance_base: float = Form(12.0),
    strength_refiner: float = Form(0.4),
    guidance_refiner: float = Form(7.5)
):
    """
    🚀 Stylizes an image using SDXL Base & Refiner.
    - Base model applies **artistic transformation**.
    - Refiner model **enhances details**.
    - Uses a **face mask** to protect facial structure.
    """
    if sdxl_base is None or sdxl_refiner is None:
        return {"status": "error", "message": "Stable Diffusion models failed to load."}

    try:
        logger.info(f"🖼️ Received request to stylize image with prompt: {prompt}")
        
        # Read and preprocess the image
        input_image = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((1024, 1024))
        
        # Generate a precise face mask
        face_mask = detect_face_mask(input_image)

        # 🎨 **Step 1: Apply artistic transformation with SDXL Base**
        logger.info("🎨 Applying SDXL Base for style transfer...")
        stylized_image = sdxl_base(
            prompt=prompt,
            image=input_image,
            strength=strength_base,
            guidance_scale=guidance_base,
            mask_image=face_mask
        ).images[0]

        # 🔍 **Step 2: Enhance details with SDXL Refiner**
        logger.info("🔍 Applying SDXL Refiner for fine-tuning...")
        refined_image = sdxl_refiner(
            prompt="Enhance details, improve color and texture",
            image=stylized_image,
            strength=strength_refiner,
            guidance_scale=guidance_refiner
        ).images[0]

        # 🖼️ Convert to bytes and return the final image
        img_io = io.BytesIO()
        refined_image.save(img_io, format="PNG")
        img_io.seek(0)

        logger.info("✅ Image successfully stylized!")
        return Response(content=img_io.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error(f"❌ Error during image stylization: {e}")
        return {"status": "error", "message": str(e)}
