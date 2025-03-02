import logging
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from segment_anything import SamPredictor, sam_model_registry

logger = logging.getLogger("AI-Styler")

def load_sd_model():
    """Loads Stable Diffusion and returns the pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
        return pipe
    except Exception as e:
        logger.error(f"Error loading Stable Diffusion model: {e}")
        return None

def load_sam_model():
    """Loads SAM and returns the predictor."""
    sam_checkpoint = "sam_vit_b.pth"  # Ensure you have the SAM checkpoint file
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        sam.to(device)
        return SamPredictor(sam)
    except Exception as e:
        logger.error(f"Error loading SAM model: {e}")
        return None
