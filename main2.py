# worker/main.py

# === Standard Library Imports ===
import sys
import os
import logging
import argparse
from contextlib import asynccontextmanager
import asyncio

# === Third-party Imports ===
import torch
import uvicorn
from fastapi import FastAPI
import openai
import httpx
from utils import start_server

# === Project Path Setup ===
WORKER_ROOT = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.join(WORKER_ROOT, "validation")
GEN3D_DIR = os.path.join(WORKER_ROOT, "generate")

for path in [WORKER_ROOT, VALIDATION_DIR, GEN3D_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# === Local Imports ===
from config import parse_arguments, get_device
from utils import setup_logging, cleanup_gpu_memory
from t2i_generator import T2IGenerator
from i23d_generator import I23DGenerator
from validator2 import Validator
from validation.engine.io.ply import PlyLoader
from validation.engine.validation_engine import ValidationEngine
from validation.engine.rendering.renderer import Renderer
from routes2 import router as api_router

logger = logging.getLogger(__name__)

# === FastAPI Lifespan Handler ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời của ứng dụng FastAPI: khởi tạo tài nguyên khi bắt đầu
    và dọn dẹp khi kết thúc.
    """
    logger.info("--- Lifespan Startup Initiated ---")

    # Khởi tạo các state của app về None để đảm bảo an toàn
    app.state.config_args = None
    app.state.t2i_generator = None
    app.state.i23d_generator = None
    app.state.validator = None
    app.state.openai_client = None
    app.state.vlm_client = None
    app.state.busy_flag_lock = asyncio.Lock()
    app.state.is_busy = False

    try:
        # 1. Parse Arguments and Setup Logging
        args = parse_arguments()
        setup_logging(args.log_level)
        logger.info(f"Parsed arguments: {vars(args)}")
        app.state.config_args = args

        # 2. Get Device
        device = get_device(args)
        logger.info(f"Using device: {device}")

        # 3. Initialize OpenAI Clients
        # Client cho các tác vụ LLM (e.g., prompt enhancement)
        logger.info(f"Initializing OpenAI client for LLM tasks (API Key ending with '...{args.api_key[-4:]}')")
        app.state.openai_client = openai.AsyncOpenAI(
            api_key=args.api_key,
            timeout=10.0 # Timeout ngắn cho các tác vụ LLM nhanh
        )

        # Client cho các tác vụ VLM (e.g., image selection, final judgement)
        logger.info(f"Initializing OpenAI-compatible client for VLM tasks (Endpoint: {args.vlm_endpoint}, API Key ending with '...{args.vlm_api_key[-4:]}')")
        app.state.vlm_client = openai.AsyncOpenAI(
            base_url=args.vlm_endpoint,
            api_key="local",
            timeout=5.0, # Timeout dài hơn cho các tác vụ VLM có thể nặng
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
        )
        logger.info("All API clients initialized.")

        # 4. Initialize Core Components (Validation Engine, Renderer, etc.)
        logger.info("Initializing validation engine and renderer...")
        validation_engine = ValidationEngine(verbose=(args.log_level.lower() == 'debug'))
        # validation_engine.load_pipelines()
        ply_loader = PlyLoader()
        renderer = Renderer()
        
        # 5. Initialize Validator (sử dụng các components trên)
        app.state.validator = Validator(
            validation_engine=validation_engine,
            ply_loader=ply_loader,
            renderer=renderer,
            device=device
        )
        logger.info("Validator component initialized.")

        # 6. Initialize Generators
        logger.info("Initializing T2I and I23D generators...")
        app.state.t2i_generator = T2IGenerator(
            api_url=args.t2i_api_url,
            device=device,
            multiple_t2i_api=args.multiple_t2i_api,
            save_images=args.save_images,
            image_size=args.image_size
        )
        app.state.i23d_generator = I23DGenerator(
            trellis_model_id=args.trellis_model_id,
            device=device,
            port=args.port,
            validate_endpoint=args.validate_endpoint
        )
        logger.info("All generators initialized.")

        logger.info("--- Lifespan Startup Completed Successfully ---")

    except Exception:
        logger.critical("CRITICAL ERROR during application startup. The server might not be functional.", exc_info=True)
        # Có thể raise exception ở đây để ngăn server khởi động nếu lỗi nghiêm trọng
        # raise

    # --- Application is now running ---
    yield
    # --- Shutdown sequence starts after this point ---

    logger.info("--- Lifespan Shutdown Initiated ---")
    try:
        # Dọn dẹp theo thứ tự ngược lại
        if hasattr(app.state, 'i23d_generator') and app.state.i23d_generator:
            app.state.i23d_generator.unload_models()
        if hasattr(app.state, 't2i_generator') and app.state.t2i_generator:
            app.state.t2i_generator.unload_model()
        if hasattr(app.state, 'validator') and app.state.validator and hasattr(app.state.validator, 'validation_engine'):
             app.state.validator.validation_engine.unload_pipelines()
        
        # Đóng các client API
        if hasattr(app.state, 'openai_client') and app.state.openai_client:
            await app.state.openai_client.close()
        if hasattr(app.state, 'vlm_client') and app.state.vlm_client:
            await app.state.vlm_client.close()

        logger.info("Models unloaded and API clients closed.")
    except Exception:
        logger.error("Error during model/client cleanup.", exc_info=True)

    cleanup_gpu_memory()
    logger.info("--- Lifespan Shutdown Completed ---")


# === FastAPI App Initialization ===
app = FastAPI(
    title="Sana Worker API - 3D Generation",
    version="4.0.0",
    description="API for multi-stage, VLM-assisted 3D model generation",
    lifespan=lifespan,
)

# Gắn router API vào ứng dụng chính
app.include_router(api_router)


# === Entry Point ===
if __name__ == "__main__":
    # Phân tích các tham số dòng lệnh một lần ở đây để uvicorn sử dụng
    # File config.py cũng cần được cập nhật để chứa vlm_endpoint và vlm_api_key
    cli_args = parse_arguments()

    logger.info(f"Starting Uvicorn server at http://{cli_args.host}:{cli_args.port}")
    start_server(cli_args.port)
    uvicorn.run(
        app,
        host=cli_args.host,
        port=cli_args.port,
        log_level=cli_args.log_level.lower(),
        reload=False, # Không nên dùng reload trong production
    )