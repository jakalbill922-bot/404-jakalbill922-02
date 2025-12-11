# worker/routes.py
import logging
import time
from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Annotated
import argparse
import os
import asyncio

# --- Local Imports ---
from models import Gen3DRequest, Gen3DResponse, GetPlyRequest, GetPlyResponse, ScoreDetail
from orchestrator2 import process_generation_request
from utils import encode_bytes_base64, cleanup_gpu_memory, _sanitize_filename
from t2i_generator import T2IGenerator
from i23d_generator import I23DGenerator
from validator2 import Validator
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Dependency Injection ---
def get_t2i_generator(request: Request) -> T2IGenerator:
    generator = request.app.state.t2i_generator
    if not generator: raise HTTPException(status_code=503, detail="T2I Generator not ready.")
    return generator

def get_i23d_generator(request: Request) -> I23DGenerator:
    generator = request.app.state.i23d_generator
    if not generator: raise HTTPException(status_code=503, detail="I23D Generator not ready.")
    return generator

def get_validator(request: Request) -> Validator:
    validator = request.app.state.validator
    if not validator: raise HTTPException(status_code=503, detail="Validator component not ready.")
    return validator

def get_config(request: Request) -> argparse.Namespace:
    config_args = request.app.state.config_args
    if not config_args: raise HTTPException(status_code=503, detail="Configuration not loaded.")
    return config_args

T2IGeneratorDep = Annotated[T2IGenerator, Depends(get_t2i_generator)]
I23DGeneratorDep = Annotated[I23DGenerator, Depends(get_i23d_generator)]
ValidatorDep = Annotated[Validator, Depends(get_validator)]
ConfigDep = Annotated[argparse.Namespace, Depends(get_config)]

# --- API Endpoints ---
@router.post("/gen3d", response_model=Gen3DResponse)
async def generate_3d_endpoint(
    request: Request,
    request_data: Gen3DRequest,
    t2i_generator: T2IGeneratorDep,
    i23d_generator: I23DGeneratorDep,
    validator: ValidatorDep,
    config_args: ConfigDep
):
    endpoint_start_time = time.time()
    logger.info("="*100)
    log_prompt = request_data.prompt if request_data.task_type == "text" else request_data.task_id
    logger.info(f"[START] Task type: {request_data.task_type} - Task ID: {request_data.task_id} - Prompt: {log_prompt}")
    os.makedirs(f"/root/save_t2i/{datetime.now().strftime('%Y-%m-%d')}", exist_ok=True)
    
    busy_flag_lock = request.app.state.busy_flag_lock
    app_state = request.app.state
    async with busy_flag_lock:
        if app_state.is_busy:
            logger.warning(f"[/gen3d] Worker is busy. Rejecting request for '{log_prompt}'.")
            return Gen3DResponse(message="BUSY", success=False, cancelled=False)
        app_state.is_busy = True
        logger.info(f"[/gen3d] Accepted request for '{log_prompt}'. Worker marked as busy.")
    
    # --- Định nghĩa hàm kiểm tra hủy bỏ ---
    async def check_cancelled() -> bool:
        # FastAPI/Starlette sẽ cập nhật trạng thái này
        return await request.is_disconnected()

    try:
        config_dict = {
            "use_edit_llm": config_args.use_edit_llm,
            "spz_level": config_args.spz_level,
            "num_candidate_images": getattr(config_args, 'num_candidate_images', 4),
            "top_k_images": getattr(config_args, 'top_k_images', 2),
            "validate_endpoint": config_args.validate_endpoint,
            "vlm_endpoint": config_args.vlm_endpoint,
            "save_images": config_args.save_images,
            "port": config_args.port,
            "task_type": request_data.task_type,
            "task_id": request_data.task_id
        }

        # *** THAY ĐỔI QUAN TRỌNG: TRUYỀN `check_cancelled` VÀO ***
        result, compressed_bytes = await process_generation_request(
            req=request_data,
            t2i_generator=t2i_generator,
            i23d_generator=i23d_generator,
            validator=validator,
            openai_client=request.app.state.openai_client,
            vlm_client=request.app.state.vlm_client,
            config=config_dict,
            cancel_check=check_cancelled # Truyền hàm vào orchestrator
        )
        
        ply_local_path = None
        ply_base64_encoded = None
        if result.get("success") and compressed_bytes:
            if request_data.return_ply:
                ply_base64_encoded = encode_bytes_base64(compressed_bytes)
            else:
                safe_fname = _sanitize_filename(request_data.prompt) + ".ply.spz"
                with open(safe_fname, "wb") as f: f.write(compressed_bytes)
                ply_local_path = safe_fname
                logger.info(f"Compressed PLY saved to: {ply_local_path}")
        
        # Cập nhật response dictionary với các giá trị vừa xử lý
        result.update({
            "ply_local_path": ply_local_path,
            "ply_base64": ply_base64_encoded
        })
        
        # Đảm bảo các trường luôn tồn tại trong response model
        response_data = {
            "success": result.get("success", False),
            "message": result.get("message", "An unknown error occurred."),
            "score": result.get("score", 0),
            "duration": result.get("duration"),
            "ply_local_path": result.get("ply_local_path"),
            "ply_base64": result.get("ply_base64"),
            "retry": result.get("retry"),
            "target_met": result.get("target_met"),
            "error": result.get("error"),
            "cancelled": result.get("cancelled", False),
            "score_detail": ScoreDetail(**result["score_detail"]) if result.get("score_detail") else None,
        }
        return Gen3DResponse(**response_data)

    except Exception as e:
        logger.exception("[/gen3d] Unhandled exception in endpoint handler.")
        is_disconnected = await check_cancelled()
        return Gen3DResponse(
            success=False, message="A critical server error occurred.",
            error=f"Internal Server Error: {type(e).__name__}",
            duration=round(time.time() - endpoint_start_time, 2),
            cancelled=is_disconnected
        )
    finally:
        async with busy_flag_lock:
            if app_state.is_busy:
                app_state.is_busy = False
                logger.info(f"[/gen3d] Request processing finished for '{log_prompt}'. Worker marked as NOT BUSY.")
        cleanup_gpu_memory()


@router.post("/get_ply", response_model=GetPlyResponse)
async def get_ply_file(request_data: GetPlyRequest):
    endpoint_start_time = time.time()
    file_path = request_data.ply_local_path

    logger.info(f"[/get_ply] Received request for path: {file_path}")
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError
        
        ### Data được nén
        with open(file_path, "rb") as f:
            ply_base64_encoded = encode_bytes_base64(f.read())

        # Xoá file sau khi phục vụ
        os.remove(file_path)
        logger.info(f"[/get_ply] Done in {time.time() - endpoint_start_time:.2f}s")
        return GetPlyResponse(success=True, ply_base64=ply_base64_encoded)

    except FileNotFoundError:
        logger.warning(f"[/get_ply] File not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    except Exception as e:
        logger.exception(f"[/get_ply] Unexpected error processing path: {file_path}")
        raise HTTPException(status_code=500, detail=f"Failed to process PLY file: {type(e).__name__}")


@router.get("/")
async def root():
    return "Worker sana ft higen3d dual is running!"
