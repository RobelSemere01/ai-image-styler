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
async def stylize_image(file: UploadFile, prompt: str = Form(...)):
    """
    Transforms an uploaded image based on a given artistic style prompt with segmentation.
    """
    try:
        logger.info("Received request to stylize image with prompt: %s", prompt)

        # Read and preprocess the image
        input_image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_array = np.array(input_image)

        # Apply segmentation
        logger.info("Applying Segment Anything Model")
        sam_predictor.set_image(image_array)
        masks, _, _ = sam_predictor.predict(
            point_coords=np.array([[image_array.shape[1] // 2, image_array.shape[0] // 2]]),  # Center point
            point_labels=np.array([1]),
            multimask_output=False
        )

        # Apply mask to the image
        mask = masks[0]  # Assuming single mask
        segmented_image = cv2.bitwise_and(image_array, image_array, mask=mask.astype(np.uint8) * 255)

        # Convert back to PIL Image
        segmented_pil_image = Image.fromarray(segmented_image).resize((512, 512))

        # Apply style transformation using Stable Diffusion
        logger.info("Applying Stable Diffusion")
        styled_image = pipe(prompt=prompt, image=segmented_pil_image, strength=0.45, guidance_scale=7.5).images[0]

        # Convert image to bytes
        img_io = io.BytesIO()
        styled_image.save(img_io, format="PNG")
        img_io.seek(0)

        logger.info("Image stylized successfully")
        return Response(content=img_io.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error("Error during stylization: %s", e)
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
