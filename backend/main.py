from fastapi import FastAPI, UploadFile, Form, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import cv2
import io
import torch
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI-Styler")

app = FastAPI()

# Enable CORS for frontend at localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Tillåt frontendens origin
    allow_credentials=True,
    allow_methods=["*"],  # Tillåt alla HTTP-metoder
    allow_headers=["*"],  # Tillåt alla headers
)

# Load pre-trained Stable Diffusion model
device = "cuda" if torch.cuda.is_available() else "cpu"  # Använd GPU om tillgänglig
logger.info(f"Using device: {device}")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

# Load Segment Anything Model (SAM)
sam_checkpoint = "sam_vit_b.pth"  # Ensure you have the SAM checkpoint file
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to(device)
sam_predictor = SamPredictor(sam)

@app.post("/stylize-image/")
async def stylize_image(file: UploadFile, 
                        prompt: str = Form(...), 
                        coordinates: str = Form(...)):  # 🆕 Tar emot koordinater från frontend
    """
    Transforms an uploaded image based on a given artistic style prompt with segmentation.
    """
    try:
        logger.info("Received request to stylize image with prompt: %s", prompt)

        # 📌 Läs in bilden
        input_image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_array = np.array(input_image)

        # 📌 Omvandla koordiner från sträng till numpy array
        coords_list = [list(map(int, coord.split(','))) for coord in coordinates.split(';')]
        point_coords = np.array(coords_list)
        point_labels = np.ones(len(point_coords))  # Markera alla punkter som positiva

        # 📌 Använd Segment Anything Model med användarens valda punkter
        logger.info(f"Applying SAM with user coordinates: {point_coords}")
        sam_predictor.set_image(image_array)
        masks, _, _ = sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False
        )

        # 📌 Skapa mask och applicera på bilden
        mask = masks[0]  # Tar den första masken
        segmented_image = cv2.bitwise_and(image_array, image_array, mask=mask.astype(np.uint8) * 255)

        # 📌 Konvertera tillbaka till PIL och kör Stable Diffusion
        segmented_pil_image = Image.fromarray(segmented_image).resize((512, 512))

        logger.info("Applying Stable Diffusion")
        styled_image = pipe(prompt=prompt, image=segmented_pil_image, strength=0.45, guidance_scale=7.5).images[0]

        # 📌 Konvertera bild till bytes och returnera
        img_io = io.BytesIO()
        styled_image.save(img_io, format="PNG")
        img_io.seek(0)

        logger.info("Image stylized successfully")
        return Response(content=img_io.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error("Error during stylization: %s", e)
        return {"status": "error", "message": str(e)}


@app.post("/preview-mask/")
async def preview_mask(file: UploadFile, coordinates: str = Form(...)):
    """
    Generates a segmentation mask based on user input before applying Stable Diffusion.
    """
    try:
        logger.info("Generating segmentation mask")

        # 📌 Läs in bilden
        input_image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_array = np.array(input_image)

        # 📌 Omvandla koordiner från sträng till numpy array
        coords_list = [list(map(int, coord.split(','))) for coord in coordinates.split(';')]
        point_coords = np.array(coords_list)
        point_labels = np.ones(len(point_coords))  # Markera alla punkter som positiva

        # 📌 Använd SAM för segmentering
        logger.info(f"Applying SAM with user coordinates: {point_coords}")
        sam_predictor.set_image(image_array)
        masks, _, _ = sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False
        )

        # 📌 Kontrollera om mask genererades
        if masks is None or len(masks) == 0:
            return {"status": "error", "message": "No mask was generated."}

        mask = masks[0]  # Tar första masken

        # 📌 Skapa en synlig mask
        mask_preview = (mask * 255).astype(np.uint8)
        mask_preview = Image.fromarray(mask_preview)

        # 📌 Konvertera masken till bytes och returnera
        img_io = io.BytesIO()
        mask_preview.save(img_io, format="PNG")
        img_io.seek(0)

        logger.info("Mask preview generated successfully")
        return Response(content=img_io.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error("Error during mask generation: %s", e)
        return {"status": "error", "message": str(e)}


@app.get("/user-history/")
def get_user_history():
    """
    Mock endpoint for user history.
    Replace this with actual database integration if required.
    """
    logger.info("Fetching user history")
    return [
        {"id": 1, "image_url": "http://localhost:8000/static/sample1.png"},
        {"id": 2, "image_url": "http://localhost:8000/static/sample2.png"},
    ]

@app.websocket("/progress/")
async def progress_websocket(websocket: WebSocket):
    logger.info("WebSocket connection established")
    await websocket.accept()
    try:
        for i in range(1, 101, 10):  # Skicka dummyprogress för test
            await websocket.send_json({"progress": i})
            logger.info("Progress sent: %d%%", i)
            await asyncio.sleep(1)
    except Exception as e:
        logger.error("WebSocket disconnected: %s", e)
    finally:
        await websocket.close()
        logger.info("WebSocket connection closed")
