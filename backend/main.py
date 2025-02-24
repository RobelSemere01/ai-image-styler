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
    allow_origins=["http://localhost:3000"],  # Allow the frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained Stable Diffusion model
device = "cuda" if torch.cuda.is_available() else "cpu"
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
                        box_coordinates: str = Form(...)):
    """
    Transforms an uploaded image based on a given artistic style prompt using a bounding box for segmentation.
    Only the region within the mask (derived from the bounding box) will be stylized,
    while the rest of the image remains unchanged.
    """
    try:
        logger.info("Received request to stylize image with prompt: %s", prompt)

        # Read and convert the image
        input_image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_array = np.array(input_image)

        # Parse the bounding box coordinates (format: "x0,y0,x1,y1")
        box_coords = list(map(int, box_coordinates.split(',')))
        box = np.array(box_coords)

        # Use SAM with the bounding box prompt to obtain the segmentation mask
        logger.info(f"Applying SAM with bounding box: {box}")
        sam_predictor.set_image(image_array)
        masks, _, _ = sam_predictor.predict(
            box=box,
            multimask_output=False
        )
        
        # Check if a mask was generated
        if masks is None or len(masks) == 0:
            return {"status": "error", "message": "No mask was generated."}

        # Convert mask to proper format (0-255)
        mask = masks[0].astype(np.uint8) * 255

        # Create a blended image using a simple alpha blend.
        # (This is used as input to the diffusion pipeline.)
        alpha = 0.5  # blending weight: 0 gives only mask, 1 gives only original
        blended_image = cv2.addWeighted(image_array, alpha, np.stack([mask]*3, axis=-1), 1 - alpha, 0)
        blended_pil_image = Image.fromarray(blended_image).resize((512, 512))

        logger.info("Applying Stable Diffusion")
        styled_image = pipe(prompt=prompt, image=blended_pil_image, strength=0.45, guidance_scale=7.5).images[0]

        # Convert the styled image and original image to numpy arrays
        styled_np = np.array(styled_image.resize(input_image.size))  # scale back to original size
        original_np = image_array

        # Create a composite image: use the stylized version only where mask is present
        # First, convert mask to a binary mask with values 0 or 1
        binary_mask = (mask / 255.0).astype(np.float32)
        # Expand dims to make it 3-channel
        binary_mask = np.stack([binary_mask]*3, axis=-1)

        # Composite: stylized region from diffusion, original for the rest.
        final_np = (styled_np * binary_mask + original_np * (1 - binary_mask)).astype(np.uint8)
        final_image = Image.fromarray(final_np)

        # Convert final image to bytes and return
        img_io = io.BytesIO()
        final_image.save(img_io, format="PNG")
        img_io.seek(0)

        logger.info("Image stylized successfully")
        return Response(content=img_io.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error("Error during stylization: %s", e)
        return {"status": "error", "message": str(e)}

@app.post("/preview-mask/")
async def preview_mask(file: UploadFile, box_coordinates: str = Form(...)):
    """
    Generates a segmentation mask based on a bounding box prompt.
    """
    try:
        logger.info("Generating segmentation mask")

        # Read and convert the image
        input_image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_array = np.array(input_image)

        # Parse the bounding box coordinates (format: "x0,y0,x1,y1")
        box_coords = list(map(int, box_coordinates.split(',')))
        box = np.array(box_coords)

        # Use SAM for segmentation with the bounding box prompt
        logger.info(f"Applying SAM with bounding box: {box}")
        sam_predictor.set_image(image_array)
        masks, _, _ = sam_predictor.predict(
            box=box,
            multimask_output=False
        )

        if masks is None or len(masks) == 0:
            return {"status": "error", "message": "No mask was generated."}

        mask = masks[0]  # Use the first mask

        # Create a visible mask preview (scale binary mask to 0-255)
        mask_preview = (mask * 255).astype(np.uint8)
        mask_preview = Image.fromarray(mask_preview)

        # Convert mask preview to bytes and return
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
        {"id": 1, "image_url": "http://localhost:8087/static/sample1.png"},
        {"id": 2, "image_url": "http://localhost:8087/static/sample2.png"},
    ]

@app.websocket("/progress/")
async def progress_websocket(websocket: WebSocket):
    logger.info("WebSocket connection established")
    await websocket.accept()
    try:
        for i in range(1, 101, 10):  # Send dummy progress for testing
            await websocket.send_json({"progress": i})
            logger.info("Progress sent: %d%%", i)
            await asyncio.sleep(1)
    except Exception as e:
        logger.error("WebSocket disconnected: %s", e)
    finally:
        await websocket.close()
        logger.info("WebSocket connection closed")
