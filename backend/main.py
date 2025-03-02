from fastapi import FastAPI
from backend.Segment import router as segment_router  # ✅ Import Segment Anything API
from backend.Transformer import router as transformer_router  # ✅ Import Image Stylization API
from backend.config import add_middlewares


app = FastAPI()
add_middlewares(app)

# ✅ Register routers
app.include_router(segment_router)  # Includes all Segment Anything routes
app.include_router(transformer_router)  # Includes all Image Stylization routes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
