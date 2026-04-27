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
    "http://10.7.40.120:8080",
]

# --- ✅ CRITICAL: Handle preflight FIRST (before any other middleware) ---
@app.middleware("http")
async def handle_cors_preflight(request: Request, call_next):
    """
    Handle CORS preflight OPTIONS requests.
    This MUST run before other middleware.
    """
    origin = request.headers.get("origin", "")
    
    # Handle OPTIONS preflight requests
    if request.method == "OPTIONS":
        logger.info(f"🔍 Preflight OPTIONS request from origin: {origin}")
        
        if origin in ALLOWED_ORIGINS or origin == "":
            return JSONResponse(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": origin or "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Max-Age": "3600",
                }
            )
        else:
            logger.warning(f"❌ Origin not allowed: {origin}")
            return JSONResponse(
                status_code=403,
                content={"detail": "Origin not allowed"}
            )
    
    # Process normal requests
    try:
        response = await call_next(request)
        
        # Add CORS headers to response
        if origin in ALLOWED_ORIGINS or origin == "":
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Expose-Headers"] = "Content-Type, Authorization"
        
        return response
    except Exception as e:
        logger.error(f"❌ Error processing request: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

# --- CORS Middleware (secondary layer) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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

# --- Catch-all OPTIONS handler ---
@app.options("/{full_path:path}", tags=["CORS"])
async def catch_all_options(full_path: str, request: Request):
    """
    Catch-all OPTIONS handler for any endpoint.
    Useful for ngrok compatibility.
    """
    origin = request.headers.get("origin", "")
    logger.info(f"🔍 Catch-all OPTIONS for /{full_path} from {origin}")
    
    if origin in ALLOWED_ORIGINS or origin == "":
        return JSONResponse(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": origin or "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Max-Age": "3600",
            }
        )
    else:
        return JSONResponse(status_code=403, content={"detail": "Origin not allowed"})

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting uvicorn server...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)