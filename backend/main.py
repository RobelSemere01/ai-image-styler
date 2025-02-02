from fastapi import FastAPI, UploadFile, Form
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import io

app = FastAPI()

# Load pre-trained Stable Diffusion model
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cpu")  # Use "cuda" if you have a GPU

@app.post("/stylize-image/")
async def stylize_image(file: UploadFile, prompt: str = Form(...)):
    """
    Transforms an uploaded image based on a given artistic style prompt.
    """
    try:
        # Read and preprocess the image
        input_image = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((512, 512))
        
        # Apply style transformation using the pre-trained model
        styled_image = pipe(prompt=prompt, image=input_image, strength=0.8, guidance_scale=7.5).images[0]
        
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
