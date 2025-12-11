# worker/orchestrator.py
from urllib.parse import urlparse
from datetime import datetime
import logging
import time
import asyncio
import io
import os
from typing import Dict, Any, List, Optional, Tuple, Callable, Awaitable
import httpx
import openai
import base64
import pybase64
from openai import AsyncOpenAI
from PIL import Image
# --- Local Imports ---
from t2i_generator import T2IGenerator
from i23d_generator import I23DGenerator
from validator2 import Validator
from models import Gen3DRequest
from utils import cleanup_gpu_memory, compress_ply_bytes
from image_selector import select_top_k_images_by_vlm, pick_image_by_vlm
from final_judgement import select_best_3d_model_by_vlm
import re
from brighten import brighten_image
from rotate2d import rotate_image
from validation.engine.data_structures import ValidationResponse

def sanitize_filename(text: str) -> str:
    text = text.strip().lower()[:200]
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_\-]", "", text)
    text = re.sub(r"_", " ", text)
    return text

logger = logging.getLogger(__name__)
MIN_RETRY_BUFFER_TIME = 7.0

# --- Exception tùy chỉnh để xử lý việc hủy bỏ ---
class ClientCancelledError(Exception):
    """Exception được raise khi client ngắt kết nối."""
    pass

class OrchestrationState:
    """Lớp theo dõi trạng thái của một request."""
    def __init__(self, req: Gen3DRequest, cancel_check: Callable[[], Awaitable[bool]]):
        self.req = req
        self.start_time = time.time()
        self.cancel_check = cancel_check
        self.candidates_3d: List[Tuple[float, bytes]] = []
        self.result: Dict[str, Any] = {"success": False, "message": "Processing started."}
        self.final_compressed_ply: Optional[bytes] = None
        self.final_compressed_base64: Optional[str] = None

    async def check_cancellation(self):
        """Kiểm tra và raise exception nếu client đã hủy."""
        if await self.cancel_check():
            raise ClientCancelledError("Client disconnected during processing.")

    def time_remaining(self) -> float:
        return self.req.timeout - (time.time() - self.start_time)

    def add_candidate(self, score: float, ply_bytes: bytes):
        self.candidates_3d.append((score, ply_bytes))
        logger.info(f"Added new 3D candidate with score {score:.3f}. Total candidates: {len(self.candidates_3d)}")

async def process_generation_request(
    req: Gen3DRequest,
    t2i_generator: T2IGenerator,
    i23d_generator: I23DGenerator,
    validator: Validator,
    openai_client: openai.AsyncOpenAI,
    vlm_client: openai.AsyncOpenAI,
    config: Dict[str, Any],
    cancel_check: Callable[[], Awaitable[bool]]
) -> Tuple[Dict[str, Any], Optional[bytes]]:
    
    state = OrchestrationState(req, cancel_check)
    logger.info(f"Orchestration started for prompt: '{req.prompt[:70]}...' with timeout {req.timeout}s")

    try:
        await state.check_cancellation()

        # --- BƯỚC 2: TẠO HOẶC TẢI ẢNH 2D ---
        t1 = time.perf_counter()
        candidate_images = []
        if req.img_paths and len(req.img_paths) > 0:
            logger.info(f"Using {len(req.img_paths)} provided image paths (skip T2I generation).")
            for p in req.img_paths:
                try:
                    if not os.path.exists(p):
                        logger.warning(f"Image path not found: {p}")
                        continue
                    img = Image.open(p)
                    candidate_images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load image from {p}: {e}")
            if not candidate_images:
                raise RuntimeError("All provided image paths are invalid or failed to load.")
        elif req.task_type == "text":
            # Trường hợp bình thường: generate ảnh từ prompt
            num_initial_images = config.get('num_candidate_images', 1)
            if req.n_t2i is not None:
                num_initial_images = req.n_t2i

            candidate_images = await t2i_generator.generate(
                prompt=req.prompt, 
                enhanced_types=["llm", "template", "raw"][:num_initial_images],
                cancel_check=state.check_cancellation,
                openai_client=openai_client,
                req_t2i_url=req.t2i_url,
                port=config.get('port', 0)
            )
            logger.info(f"[TIME T2I] Generated {len(candidate_images)} images (took {time.perf_counter() - t1:.2f}s)")
        elif req.task_type == "image":
            prompt_image = Image.open(io.BytesIO(pybase64.b64decode(req.prompt)))
            filename = f"[IMAGE]_{req.task_id}_{config.get('port', 0)}.png"
            prompt_image.save(f"/root/save_t2i/{datetime.now().strftime('%Y-%m-%d')}/{filename}")
            # rembg_image = t2i_generator.remove_background(prompt_image)
            candidate_images = [prompt_image]

        if not candidate_images:
            raise RuntimeError("Failed to obtain any 2D candidate images.")
            
        # --- BƯỚC 3: LỌC ẢNH 2D BẰNG VLM ---
        await state.check_cancellation()
        t2 = time.perf_counter()
        top_k = config.get('top_k_images', 1)

        if len(candidate_images) > 1:
            try:
                selected_images = [await asyncio.wait_for(
                    pick_image_by_vlm(req.prompt, candidate_images, vlm_client),
                    timeout=5.0
                )]
            except asyncio.TimeoutError:
                logger.warning("VLM image selection timeout. Using first top_k images.")
                selected_images = candidate_images[:top_k]
        else:
            selected_images = candidate_images
            
        logger.info(f"Selected {len(selected_images)} best 2D images (took {time.perf_counter() - t2:.2f}s)")
        if not selected_images:
            raise RuntimeError("No image selected for 3D generation.")

        # --- BƯỚC 4: TẠO 3D ---
        best_val_result = None
        for i, image in enumerate(selected_images):
            await state.check_cancellation()

            # Preprocessing nhỏ (xoay/sáng)
            safe_prompt = sanitize_filename(req.prompt) if req.task_type == "text" else req.task_id
            if req.task_type == "text":
                _, image = rotate_image(image, req.prompt.lower())
                _, image = brighten_image(image, req.prompt.lower())

                if config.get('save_images', 0):
                    image_path = f"/root/save_t2i/{datetime.now().strftime('%Y-%m-%d')}/[RMBG]_{safe_prompt}_{i}_{config.get('port', 0)}.png"
                    image.save(image_path)

            if state.time_remaining() < MIN_RETRY_BUFFER_TIME:
                logger.warning(f"Timeout imminent. Stopping further 3D generation. Time left: {state.time_remaining():.2f}s")
                break

            t3 = time.perf_counter()
            gaussians_list, gen_err = await asyncio.to_thread(
                i23d_generator.generate,
                image=image,
                do_rotate=True,
                light_mode=req.light_mode,
                prompt=safe_prompt,
                task_type=req.task_type,
                task_id=req.task_id,
                edit_image_url=req.edit_image_url
            )
            logger.info(f"I23D generation took {time.perf_counter() - t3:.2f}s")

            await state.check_cancellation()

            if gen_err or not gaussians_list:
                logger.error(f"I23D generation failed for image {i+1}: {gen_err}")
                continue

            # --- VALIDATION ---
            for j, gaussians in enumerate(gaussians_list):
                t4 = time.perf_counter()
                with io.BytesIO() as buf:
                    # gaussians.save_ply(f"/root/save_t2i/ply/{req.task_id}_{i}_{j}.ply")
                    gaussians.save_ply(buf)
                    ply_bytes = buf.getvalue()
                comp_ply_bytes = compress_ply_bytes(ply_bytes)
                base64_data = base64.b64encode(comp_ply_bytes).decode("utf-8")

                validation_request_json = {
                    "data": base64_data,
                    "compression": 2,
                }

                try:
                    async with httpx.AsyncClient(timeout=6.0) as client:
                        validate_url = config["validate_endpoint"]
                        if req.task_type == "text":
                            validate_url = os.path.join(validate_url, "validate_txt_to_3d_ply/")
                            validation_request_json["prompt"] = req.prompt
                        elif req.task_type == "image":
                            validate_url = os.path.join(validate_url, "validate_img_to_3d_ply/")
                            validation_request_json["prompt_image"] = req.prompt

                        res = await client.post(url=validate_url, json=validation_request_json)
                        res.raise_for_status()
                        val_result = ValidationResponse(**res.json())
                        logger.info(f"Validation result: {val_result}")
                except Exception as e:
                    logger.warning(f"Validation failed via API: {e}")
                    continue

                if config.get('save_images', 0):
                    try:
                        async with httpx.AsyncClient(timeout=3.0) as client:
                            render_path = os.path.join(config["validate_endpoint"], "render_duel_view/")
                            res = await client.post(
                                url=render_path,
                                json=validation_request_json
                            )
                            filename = f"[RENDER]_{safe_prompt}_{j}_{val_result.score:.3f}_{config.get('port', 0)}.png"
                            with open(f"/root/save_t2i/{datetime.now().strftime('%Y-%m-%d')}/{filename}", "wb") as f:
                                f.write(res.content)
                    except Exception as e:
                        logger.warning(f"Render duel view failed: {e}")

                await state.check_cancellation()

                if val_result.score > 0.605:
                    state.add_candidate(val_result.score, comp_ply_bytes)
                    best_val_result = val_result
                    break
                if best_val_result is None:
                    best_val_result = val_result

        # --- BƯỚC 5: KẾT LUẬN ---
        await state.check_cancellation()
        if not state.candidates_3d:
            if best_val_result is not None:
                final_score = best_val_result.score
            else:
                final_score = -1
        else:
            final_score, state.final_compressed_ply = state.candidates_3d[0]

        state.result = {
            "success": final_score > 0.6,
            "message": "Successfully generated and selected a 3D model.",
            "score": final_score,
            "retry": len(state.candidates_3d),
            "target_met": final_score >= req.target,
            "score_detail": {
                "final_score": best_val_result.score if best_val_result else None,
                "iqa_score": best_val_result.iqa if best_val_result else None,
                "alignment_score": best_val_result.alignment if best_val_result else None,
                "ssim_score": best_val_result.ssim if best_val_result else None,
                "lpips_score": best_val_result.lpips if best_val_result else None,
            }
        }

    except ClientCancelledError as e:
        logger.warning(f"Orchestration cancelled by client: {e}")
        state.result.update({"success": False, "message": "Request cancelled by client.", "cancelled": True})
    except Exception as e:
        logger.exception("Unhandled error during orchestration.")
        state.result.update({"success": False, "message": "Orchestration failed.", "error": str(e)})
    finally:
        state.result['duration'] = round(time.time() - state.start_time, 2)
        if 'cancelled' not in state.result:
            state.result['cancelled'] = await cancel_check()
        logger.info(f"Orchestration finished in {state.result['duration']}s. Success: {state.result.get('success', False)}")
        return state.result, state.final_compressed_ply