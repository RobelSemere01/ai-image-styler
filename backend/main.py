from fastapi import FastAPI, UploadFile, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import io

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific domains if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

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
        styled_image = pipe(prompt=prompt, image=input_image, strength=0.65, guidance_scale=7.5).images[0]

        # Convert image to bytes
        img_io = io.BytesIO()
        styled_image.save(img_io, format="PNG")
        img_io.seek(0)

        return Response(content=img_io.getvalue(), media_type="image/png")

    except Exception as e:
        return {"status": "error", "message": str(e)}
