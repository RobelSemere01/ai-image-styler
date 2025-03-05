from fastapi import APIRouter, UploadFile, Form, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from backend.model import load_sd_model
from backend.model import load_sam_model
import numpy as np
import cv2
import io
import torch
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI-Styler")



router = APIRouter()  # ✅ Define router



# Load the model once for use in this route module
sdxl_base, sdxl_refiner = load_sd_model()

# Load Segment Anything Model (SAM)
sam_predictor = load_sam_model()  # ✅ Now it's stored and can be used


@router.post("/stylize-imageOld/") #gamla bara ändrad endpoint för testtning
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
        styled_image = pipe(prompt=prompt, image=blended_pil_image, strength=0.65, guidance_scale=7.5).images[0]

        # Convert the styled image and original image to numpy arrays
        styled_np = np.array(styled_image.resize(input_image.size))  # scale back to original size
        original_np = image_array

        # Create a composite image: use the stylized version only where mask is present
        # First, convert mask to a binary mask with values 0 or 1
        binary_mask = 1 - (mask / 255.0).astype(np.float32)
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

@router.post("/preview-mask/")
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

@router.get("/user-history/")
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

@router.websocket("/progress/")
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


@router.post("/stylize-image/")
async def stylize_image(file: UploadFile, 
                        prompt: str = Form(...), 
                        box_coordinates: str = Form(...),
                        strength_base: float = Form(0.7), 
                        guidance_base: float = Form(12.0),
                        strength_refiner: float = Form(0.4),
                        guidance_refiner: float = Form(7.5)):
    """
    🚀 Stylizes an uploaded image using SDXL Base & Refiner while keeping SAM segmentation logic intact.
    - Uses SAM for object segmentation.
    - Applies artistic transformation ONLY within the mask.
    - Uses SDXL Base for main transformation.
    - Uses SDXL Refiner for fine details.
    """
    if sdxl_base is None or sdxl_refiner is None:
        return {"status": "error", "message": "Stable Diffusion models failed to load."}

    try:
        logger.info(f"🖼️ Received request to stylize image with prompt: {prompt}")

        # Read and convert the image
        input_image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_array = np.array(input_image)

        # Parse the bounding box coordinates (format: "x0,y0,x1,y1")
        box_coords = list(map(int, box_coordinates.split(',')))
        box = np.array(box_coords)

        # Use SAM to generate the segmentation mask
        logger.info(f"🔹 Applying SAM with bounding box: {box}")
        sam_predictor.set_image(image_array)
        masks, _, _ = sam_predictor.predict(
            box=box,
            multimask_output=False
        )

        if masks is None or len(masks) == 0:
            return {"status": "error", "message": "No mask was generated."}

        # Convert mask to 0-255 format
        mask = masks[0].astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask)

        # ✅ Keep Blended Image (used as input to SDXL)
        alpha = 0.5  # Blending weight for mask
        blended_image = cv2.addWeighted(image_array, alpha, np.stack([mask]*3, axis=-1), 1 - alpha, 0)
        blended_pil_image = Image.fromarray(blended_image).resize((1024, 1024))  # Resize to SDXL resolution

        # 🎨 **Step 1: Apply Artistic Transformation with SDXL Base**
        logger.info("🎨 Applying SDXL Base for style transfer...")
        stylized_image = sdxl_base(
            prompt=prompt,
            image=blended_pil_image,  # Use the masked image for transformation
            strength=strength_base,
            guidance_scale=guidance_base,
            mask_image=mask_pil  # ✅ Ensures only the segmented area is transformed
        ).images[0]

        # 🔍 **Step 2: Enhance Details with SDXL Refiner**
        logger.info("🔍 Applying SDXL Refiner for fine-tuning...")
        refined_image = sdxl_refiner(
            prompt="Enhance details, improve lighting, increase texture realism",
            image=stylized_image,
            strength=strength_refiner,
            guidance_scale=guidance_refiner
        ).images[0]

        # ✅ Ensure Image Output is Valid
        if not isinstance(refined_image, Image.Image):
            logger.error("❌ SDXL Refiner returned an invalid format.")
            return {"status": "error", "message": "SDXL Refiner failed to generate a valid image."}

        # ✅ Resize back to original dimensions before compositing
        refined_np = np.array(refined_image.resize(input_image.size))
        original_np = image_array

        # ✅ Convert mask to binary format
        binary_mask = 1 - (mask / 255.0).astype(np.float32)
        binary_mask = np.stack([binary_mask]*3, axis=-1)

        # ✅ Composite: Use stylized version where the mask is present
        final_np = (refined_np * binary_mask + original_np * (1 - binary_mask)).astype(np.uint8)
        final_image = Image.fromarray(final_np)

        # ✅ Convert final image to bytes and return
        img_io = io.BytesIO()
        final_image.save(img_io, format="PNG")
        img_io.seek(0)

        logger.info("✅ Image stylized successfully!")
        return Response(content=img_io.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error(f"❌ Error during image stylization: {e}")
        return {"status": "error", "message": str(e)}

