#!/usr/bin/env python3
"""
Image-to-3D Generation Service
Combines the simplicity of serve.py with the robustness of routes2.py
"""

import gc
import argparse
import asyncio
import os
import logging
import time
from io import BytesIO
from typing import Optional
from PIL import Image

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
import numpy as np
from i23d_generator import I23DGenerator
from utils.gaussian_processor import GaussianProcessor
from plyfile import PlyData, PlyElement
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Set attention backend for optimal performance
os.environ['ATTN_BACKEND'] = 'flash-attn'

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Image-to-3D Generator API",
    version="2.0.0",
    description="Robust API for generating 3D models from images with cancellation support"
)

# Global app reference for accessing state without Request parameter
_global_app: Optional[FastAPI] = None

# ============================================================================
# Application State
# ============================================================================

class AppState:
    """Application state container"""
    def __init__(self):
        self.i23d_generator: Optional[I23DGenerator] = None
        self.is_busy: bool = False
        self.busy_flag_lock: asyncio.Lock = None
        self.config_args: Optional[argparse.Namespace] = None

# ============================================================================
# Dependency Injection
# ============================================================================

def get_app_state() -> AppState:
    """Get app state from global app reference"""
    if _global_app is None:
        raise HTTPException(status_code=503, detail="Application not initialized.")
    return _global_app.state.app_state

def get_i23d_generator() -> I23DGenerator:
    """Get I23D generator from app state"""
    app_state = get_app_state()
    generator = app_state.i23d_generator
    if not generator:
        raise HTTPException(status_code=503, detail="I23D Generator not ready.")
    return generator

# ============================================================================
# Helper Functions
# ============================================================================

def clean_vram() -> None:
    """Clean GPU VRAM"""
    gc.collect()
    torch.cuda.empty_cache()

def gaussian_to_ply_buffer(gaussian) -> BytesIO:
    """
    Convert Gaussian object to PLY format buffer
    
    Args:
        gaussian: Gaussian 3D object
        
    Returns:
        BytesIO buffer containing PLY file
    """
    gaussian_copy = gaussian.copy()
    
    # Apply flip rotation to fix up/down orientation
    # Rotate 180 degrees around X-axis to flip top/bottom
    T_flip = np.array([0, 0, 0], dtype=np.float32)
    R_flip_obj = Rotation.from_euler('xyz', [180.0, 0.0, 0.0], degrees=True)
    R_flip = R_flip_obj.as_matrix().astype(np.float32)
    
    logger.info("Applying final orientation correction (flip top/bottom)...")
    gaussian_copy.transform_data(T_flip, R_flip)

    buffer = BytesIO()
    
    # Save to buffer
    gaussian_copy.save_ply(buffer)
    buffer.seek(0)
    
    return buffer

async def process_image_to_3d(
    image: Image.Image,
    generator: I23DGenerator,
    cancel_check,
    seed: int = 0
) -> BytesIO:
    """
    Process image to 3D model with cancellation support
    
    Args:
        image: Input PIL Image
        generator: I23D Generator instance
        cancel_check: Async function to check cancellation
        seed: Random seed for generation (default: 0)
        
    Returns:
        BytesIO buffer containing PLY file
        
    Raises:
        RuntimeError: If cancelled or generation fails
    """
    t_start = time.time()
    
    # Create save directory for intermediate images
    # from datetime import datetime
    # save_dir = f"/root/save_t2i/{datetime.now().strftime('%Y-%m-%d')}"
    # os.makedirs(save_dir, exist_ok=True)
    
    # Check cancellation before starting
    if await cancel_check():
        raise RuntimeError("Request cancelled before generation started")
    
    logger.info(f"Starting 3D generation from image (seed={seed})...")
    
    # Generate 3D models (returns list of Gaussian objects)
    # Note: I23DGenerator.generate is synchronous, we run it in executor
    loop = asyncio.get_running_loop()
    
    # Wrap the synchronous generation in executor
    gaussians, error_message = await loop.run_in_executor(
        None,
        generator.generate,
        image,
        True,  # do_rotate
        0,      # light_mode
        "",     # prompt
        "image", # task_type
        "",     # task_id
        "",     # edit_image_url
        seed    # seed
    )
    
    # Check cancellation after generation
    if await cancel_check():
        clean_vram()
        raise RuntimeError("Request cancelled after generation")
    
    if error_message:
        raise RuntimeError(f"Generation failed: {error_message}")
    
    if not gaussians or len(gaussians) == 0:
        raise RuntimeError("No Gaussian objects generated")
    
    # Take the first (best) gaussian
    best_gaussian = gaussians[0]
    logger.info(f"Generated Gaussian with {best_gaussian._xyz.shape[0]} points")

    temp_ply = "temp_before_refine.ply"
    best_gaussian.save_ply(temp_ply)
    
    # Load point cloud with Open3D
    pcd = o3d.io.read_point_cloud(temp_ply)
    num_points_before = len(pcd.points)
    
    # Apply statistical outlier removal if we have enough points
    logger.warning(f"Refining PLY file with Open3D: {temp_ply}")
    
    # Statistical outlier removal - parameters from config
    # nb_neighbors: number of neighbors to consider (higher = smoother)
    # std_ratio: threshold (higher = keeps more points, preserves geometry)
    pcd_filtered, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )
    
    num_points_after = len(pcd_filtered.points)
    removed_points = num_points_before - num_points_after
    
    logger.warning(f"Outlier removal: {removed_points} points removed ({removed_points/num_points_before*100:.1f}%)")
    
    # Convert inlier_indices to numpy array for filtering
    inlier_mask = np.array(inlier_indices)
    
    # Load original PLY with plyfile to preserve all Gaussian Splatting properties
    plydata = PlyData.read(temp_ply)
    vertex = plydata['vertex']
    
    # Filter vertices using inlier mask
    filtered_vertex = vertex[inlier_mask]
    
    # Create new PLY with filtered data
    refined_ply = temp_ply.replace('.ply', '_refined.ply')
    new_vertex = PlyElement.describe(filtered_vertex, 'vertex')
    PlyData([new_vertex], text=False).write(refined_ply)
    
    logger.warning(f"Refined PLY saved with {num_points_after} points (preserved all Gaussian Splatting properties)")
    
    # Load the refined PLY back into a Gaussian object
    # best_gaussian.load_ply(refined_ply)
    logger.warning(f"Refined PLY loaded back into Gaussian object")
    
    # Clean up temporary files
    if os.path.exists(temp_ply):
        os.remove(temp_ply)
    if os.path.exists(refined_ply):
        os.remove(refined_ply)

    # Convert to PLY buffer
    buffer = gaussian_to_ply_buffer(best_gaussian)
    
    t_end = time.time()
    logger.info(f"Total processing time: {(t_end - t_start):.2f}s")
    
    # Clean up
    clean_vram()
    
    return buffer

# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/generate")
async def generate_model(
    prompt_image_file: UploadFile = File(...),
    seed: int = Form(-1)
) -> StreamingResponse:
    """
    Generate a 3D model from an uploaded image.
    
    This endpoint processes one request at a time using a busy flag.
    Supports request cancellation if client disconnects.
    
    Args:
        prompt_image_file: Uploaded image file
        seed: Random seed for generation (-1 for default/random)
        
    Returns:
        StreamingResponse with PLY file
        
    Raises:
        HTTPException: If service is busy or generation fails
    """
    endpoint_start_time = time.time()
    logger.info("="*80)
    logger.info(f"[START] /generate - File: {prompt_image_file.filename}, Seed: {seed}")
    
    # Get app state and generator
    app_state = get_app_state()
    generator = get_i23d_generator()
    busy_flag_lock = app_state.busy_flag_lock
    
    # ---- Busy Flag Check ----
    async with busy_flag_lock:
        if app_state.is_busy:
            logger.warning(f"[/generate] Worker is busy. Rejecting request.")
            raise HTTPException(status_code=503, detail="Service is busy processing another request")
        app_state.is_busy = True
        logger.info(f"[/generate] Request accepted. Worker marked as BUSY.")
    
    # ---- Define Cancellation Check ----
    # Note: Competition requires sequential handling, cancellation check is optional
    async def check_cancelled() -> bool:
        """Check if client has disconnected"""
        return False  # Simplified for competition requirements
    
    try:
        # Read and validate image
        logger.info(f"Reading image file: {prompt_image_file.filename}")
        contents = await prompt_image_file.read()
        
        if await check_cancelled():
            logger.warning("[/generate] Request cancelled during image reading")
            raise HTTPException(status_code=499, detail="Client disconnected")
        
        try:
            prompt_image = Image.open(BytesIO(contents))
            logger.info(f"Image loaded: {prompt_image.size}, mode: {prompt_image.mode}")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Use default seed if -1
        generation_seed = 0 if seed == -1 else seed
        
        # Process image to 3D
        buffer = await process_image_to_3d(
            image=prompt_image,
            generator=generator,
            cancel_check=check_cancelled,
            seed=generation_seed
        )
        
        if await check_cancelled():
            logger.warning("[/generate] Request cancelled after processing")
            raise HTTPException(status_code=499, detail="Client disconnected")
        
        # Calculate final metrics
        duration = time.time() - endpoint_start_time
        logger.info(f"[SUCCESS] Request completed in {duration:.2f}s")
        
        # Return PLY file
        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=model_{int(time.time())}.ply",
                "X-Processing-Time": f"{duration:.2f}s"
            }
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except RuntimeError as e:
        # Handle cancellation and generation errors
        error_msg = str(e)
        if "cancelled" in error_msg.lower():
            logger.warning(f"[/generate] Request cancelled: {error_msg}")
            raise HTTPException(status_code=499, detail="Request cancelled")
        else:
            logger.error(f"[/generate] Generation failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
    
    except Exception as e:
        # Handle unexpected errors
        logger.exception("[/generate] Unexpected error occurred")
        is_disconnected = await check_cancelled()
        if is_disconnected:
            raise HTTPException(status_code=499, detail="Client disconnected")
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {type(e).__name__}"
            )
    
    finally:
        # ---- Release Busy Flag ----
        async with busy_flag_lock:
            if app_state.is_busy:
                app_state.is_busy = False
                logger.info(f"[/generate] Worker marked as NOT BUSY.")
        
        # Final cleanup
        clean_vram()
        logger.info("="*80)

@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint
    
    Returns service status
    """
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Image-to-3D Generator",
        "version": "2.0.0",
        "status": "running"
    }

# ============================================================================
# Lifecycle Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models and application state on startup"""
    global _global_app
    _global_app = app
    
    logger.info("="*80)
    logger.info("Starting Image-to-3D Generation Service...")
    
    # Parse arguments
    args = get_args()
    
    # Initialize app state
    app_state = AppState()
    app_state.config_args = args
    app_state.busy_flag_lock = asyncio.Lock()
    
    # Store in FastAPI app state
    app.state.app_state = app_state
    
    logger.info(f"Configuration: {vars(args)}")
    logger.info("Loading I23D Generator models...")
    
    try:
        # Initialize I23D Generator
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        app_state.i23d_generator = I23DGenerator(
            trellis_model_id=args.model,
            device=device,
            port=args.port,
            validate_endpoint=args.validate_endpoint
        )
        
        logger.info("✓ Models loaded successfully")
        logger.info(f"✓ Server ready on http://{args.host}:{args.port}")
        logger.info("="*80)
        
    except Exception as e:
        logger.exception("Failed to load models during startup")
        raise RuntimeError("Startup failed") from e

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("="*80)
    logger.info("Shutting down service...")
    
    app_state: AppState = app.state.app_state
    
    if app_state.i23d_generator:
        logger.info("Unloading models...")
        app_state.i23d_generator.unload_models()
    
    clean_vram()
    logger.info("✓ Cleanup completed")
    logger.info("="*80)

# ============================================================================
# Command Line Arguments
# ============================================================================

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Image-to-3D Generation Server with Robust Architecture"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=10006,
        help="Server port (default: 8094)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="Stable-X/trellis-vggt-v0-2",
        help="Trellis model ID from Hugging Face (default: Stable-X/trellis-vggt-v0-2)"
    )
    
    parser.add_argument(
        "--validate-endpoint",
        type=str,
        default="",
        help="Validation endpoint URL (optional)"
    )
    
    return parser.parse_args()

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    args = get_args()
    
    logger.info(f"Starting server on http://{args.host}:{args.port}")
    logger.info(f"Using Trellis model: {args.model}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
        log_level="info"
    )

