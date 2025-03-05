from fastapi import APIRouter, UploadFile, Form, Response
import io
from PIL import Image
import logging
import numpy as np
import cv2
import torch
import mediapipe as mp
from backend.model import load_sd_model, apply_controlnet  # ✅ Ensure ControlNet is available

logger = logging.getLogger("AI-Styler")
router = APIRouter()

# ✅ Load both SDXL Base & Refiner from model.py
sdxl_base, sdxl_refiner = load_sd_model()

def detect_face_mask(image):
    """ Detects faces using MediaPipe and generates a softer mask for facial protection. """
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
            # Expand mask slightly to cover only facial features, not entire head
            cv2.rectangle(mask, (x, y), (x + width, y + height), 255, -1)

    # Refine mask for a natural blend
    mask = refine_mask(mask)
    return Image.fromarray(mask)

def refine_mask(mask):
    """Refines the mask using morphological operations for smoother transitions."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    mask = cv2.dilate(mask, kernel, iterations=2)  # Expand mask slightly
    mask = cv2.erode(mask, kernel, iterations=1)   # Shrink mask back
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
    🚀 Stylizes an image using SDXL Base & Refiner with ControlNet for better transformations.
    - Base model applies **artistic transformation**.
    - Refiner model **enhances details**.
    - Uses a **face mask** to protect facial structure but allows background changes.
    - Uses **ControlNet (Depth) to ensure background is fully modified**.
    """
    if sdxl_base is None or sdxl_refiner is None:
        return {"status": "error", "message": "Stable Diffusion models failed to load."}

    try:
        logger.info(f"🖼️ Received request to stylize image with prompt: {prompt}")
        
        # Read and preprocess the image
        input_image = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((1024, 1024))

        # Generate a **softer face mask** (only facial features, not entire head)
        refined_face_mask = detect_face_mask(input_image)

        # 🛠️ **Step 1: Apply ControlNet for structure retention**
        logger.info("🛠️ Applying ControlNet (Depth) for better background transformation...")
        controlnet_applied = apply_controlnet(input_image, mode="depth")

        # 🎨 **Step 2: Apply artistic transformation with SDXL Base**
        logger.info("🎨 Applying SDXL Base for style transfer...")
        stylized_image = sdxl_base(
            prompt=f"{prompt}, highly detailed cinematic background, 8K",
            image=controlnet_applied,
            strength=strength_base,
            guidance_scale=guidance_base,
            mask_image=refined_face_mask  # 👌 Allows controlled face preservation
        ).images[0]

        # 🔍 **Step 3: Enhance details with SDXL Refiner**
        logger.info("🔍 Applying SDXL Refiner for fine-tuning...")
        refined_image = sdxl_refiner(
            prompt="Enhance details, improve lighting, increase texture realism",
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
