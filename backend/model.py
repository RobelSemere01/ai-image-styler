import logging
import torch
from segment_anything import SamPredictor, sam_model_registry
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel

logger = logging.getLogger("AI-Styler")

# ✅ Define global variables for lazy loading
_sdxl_base = None
_sdxl_refiner = None
_controlnet = None

def load_sd_model():
    """
    Lazily loads SDXL Base & Refiner models to prevent duplicate loading.
    Returns:
        sdxl_base: Base model for strong style transfer.
        sdxl_refiner: Refiner model for detail enhancement.
    """
    global _sdxl_base, _sdxl_refiner

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if _sdxl_base is None:
        logger.info("🔄 Loading SDXL Base model...")
        _sdxl_base = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        logger.info("✅ SDXL Base model loaded successfully!")

    if _sdxl_refiner is None:
        logger.info("🔄 Loading SDXL Refiner model...")
        _sdxl_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        logger.info("✅ SDXL Refiner model loaded successfully!")

    return _sdxl_base, _sdxl_refiner

def apply_controlnet(image, mode="depth"):
    """Applies ControlNet Depth model and ensures it returns a valid image."""
    global _controlnet
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if _controlnet is None:
        logger.info("🔄 Loading ControlNet Depth model...")
        try:
            _controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth",
                torch_dtype=torch.float32  # ✅ Ensure CPU compatibility
            ).to(device)
        except Exception as e:
            logger.error(f"❌ Failed to load ControlNet: {e}")
            return image  # ✅ Return the original image if ControlNet fails

    try:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=_controlnet,
            torch_dtype=torch.float32
        ).to(device)

        result = pipe(prompt="Depth-enhanced structured image", image=image, strength=0.8).images[0]
        return result if isinstance(result, Image.Image) else image  # ✅ Ensure result is an image

    except Exception as e:
        logger.error(f"❌ ControlNet processing failed: {e}")
        return image  # ✅ Return original image instead of None




def load_sam_model():
    """Loads SAM (Segment Anything Model) and returns the predictor."""
    sam_checkpoint = "sam_vit_b.pth"  # Ensure you have the SAM checkpoint file
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        sam.to(device)
        return SamPredictor(sam)
    except Exception as e:
        logger.error(f"Error loading SAM model: {e}")
        return None
