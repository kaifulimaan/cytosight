import os
import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer

# Import local modules
from app.api import auth, upload, diagnosis, segmentation, explainability
from app.config import settings
from app.database.supabase_client import initialize_storage
from app.models.disease_model import load_model

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup logic"""
    print("\n" + "="*80)
    print("🛠️  STARTING CYTOSIGHT API SERVER")
    print("="*80)
    
    try:
        initialize_storage()
        print("   ✅ Supabase storage initialized")
    except Exception as e:
        print(f"   ⚠️  Supabase error: {e}")
    
    app.state.model = None
    print("\n✅ STARTUP COMPLETE - Server ready")
    yield
    print("\n🛑 Shutting down CytoSight API...")

# --- FastAPI App ---
app = FastAPI(
    title="CytoSight API",
    version="1.0.0",
    lifespan=lifespan
)

# --- Allowed Origins ---
ALLOWED_ORIGINS = [
    "https://cytosight.lovable.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
]

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.lovable\.app|https://.*\.lovableproject\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Authorization"],
)

# --- Include Routers ---
logger.info("📡 Registering API routers...")
app.include_router(auth.router, prefix="/api", tags=["Authentication"])
app.include_router(upload.router, prefix="/api", tags=["Storage"])
app.include_router(diagnosis.router, prefix="/api", tags=["Medical Diagnosis"])
app.include_router(segmentation.router, prefix="/api", tags=["Image Segmentation"])
app.include_router(explainability.router, prefix="/api", tags=["Explainability"])
logger.info("✅ All routers registered")

# --- Health Check ---
@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    logger.info("🏥 Health check requested")
    return {
        "message": "CytoSight API is running",
        "status": "healthy",
        "version": "1.0.0"
    }



if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting uvicorn server...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)