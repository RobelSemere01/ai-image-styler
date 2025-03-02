from fastapi import APIRouter, UploadFile, Form, Response
import io
from PIL import Image
import logging
import numpy as np
import cv2
from backend.model import load_sd_model
import os


logger = logging.getLogger("AI-Styler")
router = APIRouter()

# Load the model once for use in this route module
sd_pipe = load_sd_model()

def detect_face_mask(image):
    """ Detects face using OpenCV and generates a more precise mask. """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

# Get the absolute path of the cascade file
    cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")

    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    mask = np.zeros(image.size[::-1], dtype=np.uint8)  # Black background
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)  

    return Image.fromarray(mask)

@router.post("/stylize-imageStable/")  # ✅ Fix: Use `router.post`, not `app.post`
async def stylize_image(file: UploadFile, prompt: str = Form(...), strength: float = Form(0.4), guidance: float = Form(8.0)):
    """
    Stylizes an uploaded image using Stable Diffusion while preserving facial structure.
    """
    if sd_pipe is None:
        return {"status": "error", "message": "Stable Diffusion model failed to load."}

    try:
        logger.info(f"Received request to stylize image with prompt: {prompt}, strength: {strength}, guidance: {guidance}")

        input_image = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((512, 512))
        
        # Generate a more accurate face mask
        face_mask = detect_face_mask(input_image)

        # Apply style transfer while preserving face structure
        styled_image = sd_pipe(
            prompt=prompt,
            image=input_image,
            strength=strength,
            guidance_scale=guidance,
            mask_image=face_mask
        ).images[0]

        img_io = io.BytesIO()
        styled_image.save(img_io, format="PNG")
        img_io.seek(0)

        logger.info("Image stylized successfully with face preservation.")
        return Response(content=img_io.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error(f"Error during image stylization: {e}")
        return {"status": "error", "message": str(e)}
