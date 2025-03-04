import logging
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline  # ✅ Load SDXL Refiner
from segment_anything import SamPredictor, sam_model_registry

logger = logging.getLogger("AI-Styler")

# Define global variables (lazy loading)
_sdxl_base = None
_sdxl_refiner = None

def load_sd_model():
    """
    Lazily loads SDXL Base & Refiner models to prevent duplicate loading.
    Returns:
        sdxl_base: Base model for strong style transfer.
        sdxl_refiner: Refiner model for detail enhancement.
    """
    global _sdxl_base, _sdxl_refiner

    if _sdxl_base is None:
        logger.info("🔄 Loading SDXL Base model...")
        _sdxl_base = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32
        ).to("cpu")
        logger.info("✅ SDXL Base model loaded successfully!")

    if _sdxl_refiner is None:
        logger.info("🔄 Loading SDXL Refiner model...")
        _sdxl_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float32
        ).to("cpu")
        logger.info("✅ SDXL Refiner model loaded successfully!")

    return _sdxl_base, _sdxl_refiner


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
