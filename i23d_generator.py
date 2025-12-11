import time
import os
from datetime import datetime
import random
import asyncio
import base64
import torch
from PIL import Image
from typing import List, Tuple, Optional
import logging
import gc
import requests
import io
from rmbg_birefnet import BackgroundRemover
from utils.gaussian_processor import GaussianProcessor
from trellis.representations.gaussian import Gaussian

logger = logging.getLogger(__name__)

class I23DGenerator:
    """
    Chịu trách nhiệm tạo model 3D từ ảnh (Image-to-3D).
    """
    def __init__(self, trellis_model_id: str, device: torch.device, port: int = 0, validate_endpoint: str = ""):
        self.device = device
        self.port = port
        self.rembg_model: Optional[BackgroundRemover] = None
        self.gaussian_processor: Optional[GaussianProcessor] = None
        self.validate_endpoint = validate_endpoint
        logger.info(f"Initializing I23DGenerator on device: {self.device}")
        self._load_models(trellis_model_id)
        if not self.rembg_model:
            self.rembg_model = BackgroundRemover(device=str(self.device))

    def _load_models(self, trellis_model_id: str):
        """Tải các mô hình cần thiết cho I23D."""
        load_start_time = time.time()
        try:
            # logger.info("Loading Background Remover model...")
            # self.rembg_model = BackgroundRemover(device=str(self.device))
            # logger.info("Background Remover model loaded.")

            logger.info("Initializing GaussianProcessor and preloading Trellis...")
            self.gaussian_processor = GaussianProcessor()
            self.gaussian_processor.preload_model(trellis_model_id)
            logger.info("GaussianProcessor initialized.")
            logger.info(f"I23D models loaded in {time.time() - load_start_time:.2f} seconds.")
        except Exception as e:
            logger.exception("Failed to load I23D models.")
            raise RuntimeError("I23D models loading failed.") from e

    def generate(
        self,
        image: Image.Image,
        do_rotate: bool = False,
        light_mode: int = 0,
        prompt: str = "",
        task_type: str = "text",
        task_id: str = "",
        edit_image_url: str = "",
        seed: int = 0,
    ) -> Tuple[Optional[List[Gaussian]], str]:
        """
        Tạo ra một danh sách các đối tượng Gaussian 3D từ một ảnh 2D.
        """
        error_message: str = ""
        gaussians: Optional[List[Gaussian]] = None

        try:
            # 0. Thêm background trắng nếu ảnh là RGBA
            if image.getbands() == 4:
                # Create white background with same size
                white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
                # Composite the image onto white background
                image = Image.alpha_composite(white_bg, image.convert("RGBA"))

            # 2. Tạo 3D
            seed_3d = seed
            gaussians = None
            rotate_list = [(0,20,0)]
            if task_type == "text":
                rotate_list.extend([(0, 70, 0), (15, 0, 0), (0, 0, 15)])
            else:
                rotate_list.extend([(0, 0, -90), (-90, 0, 0), (0, 0, 180)])
            
            images = [image]
            if task_type == "image":
                if edit_image_url is not None and edit_image_url:
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    try:
                        t0 = time.time()
                        response = requests.post(
                            os.path.join(edit_image_url, "edit"),
                            json={"prompt_image": image_base64, "edit_flows": ["Generate the back view of the subject(s) in the original image. Ensure realistic lighting and consistent perspective with the original image. Keep original ratio objects"]},
                            params={"size": 512},
                            timeout=10
                        )
                        t1 = time.time()
                        print(f"[EDIT IMAGE] Time taken: {t1 - t0:0.2f} seconds")
                        response_images = response.json().get("base64_images", [])
                        new_images = []
                        for img in response_images:
                            img = Image.open(io.BytesIO(base64.b64decode(img)))
                            if len(img.getbands()) == 4:
                                white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                                img = Image.alpha_composite(white_bg, img.convert("RGBA"))
                            new_images.append(img)
                        images.extend(new_images)
                    except requests.exceptions.RequestException as e:
                        logger.error(f"Error in Qwen edit: {e}", exc_info=True)
                    except Exception as e:
                        logger.error(f"Error in Qwen edit: {e}", exc_info=True)


                images = [self.rembg_model(image) for image in images]
                if task_type == "image" and len(images) > 1:
                    # Compare images
                    base64_images = []
                    for i, image in enumerate(images):
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        base64_images.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
                    response = requests.post(
                        os.path.join(self.validate_endpoint, "compare_images"),
                        json={
                            "base64_image1": base64_images[0],
                            "base64_image2": base64_images[1],
                        },
                        timeout=3
                    )
                    alignment = response.json().get("alignment", 0.0)
                    if alignment < 0.6:
                        images = [images[0]]

                # for j, image in enumerate(images):
                #     image.save(f"/root/save_t2i/{datetime.now().strftime('%Y-%m-%d')}/[RMBG]_{task_id}_{self.port}_{j}.png")

            if light_mode:
                gaussians = self.gaussian_processor.get_model_object(
                    images=images,
                    seed=seed_3d,
                    preprocess_image=True,
                    sparse_structure_sampler_params={"steps": 30, "cfg_strength": 7.5},
                    slat_sampler_params={"steps": 12, "cfg_strength": 3},
                    additional_euler_angles_list=rotate_list,
                    rotate=do_rotate,
                    prompt=prompt
                )
            else:
                gaussians = self.gaussian_processor.get_model_object(
                    images=images,
                    seed=seed_3d,
                    preprocess_image=True,
                    sparse_structure_sampler_params={"steps": 30, "cfg_strength": 7.5},
                    slat_sampler_params={"steps": 25, "cfg_strength": 3},
                    additional_euler_angles_list=rotate_list,
                    rotate=do_rotate,
                    prompt=prompt
                )
            if not gaussians:
                raise RuntimeError("Gaussian processor failed to create a 3D model.")
            
            logger.info(f"Successfully generated 3D model with {len(gaussians)} candidates from the source image.")

        except Exception as e:
            logger.error(f"Error in I23D generation: {e}", exc_info=True)
            error_message = f"I23D Generation Error: {type(e).__name__}"
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        return gaussians, error_message
    
    def generate_from_multi_view(
        self,
        images: List[Image.Image],
        do_rotate: bool = False,
        light_mode: int = 0,
        prompt: str = "",
    ) -> Tuple[Optional[List[Gaussian]], str]:
        ## Tạo danh sách các đối tượng Gaussian 3D từ nhiều ảnh 2D.
        """
        """     
        error_message: str = ""
        gaussians: Optional[List[Gaussian]] = None

        try:
            # 1. Xóa nền nếu không phải RGBA
            for i, img in enumerate(images):
                if img.mode != "RGBA":
                    if not self.rembg_model:
                        self.rembg_model = BackgroundRemover(device=str(self.device))
                    images[i] = self.rembg_model(img)
            

            # 2. Tạo 3D
            seed_3d = 0
            gaussians = None
            rotate_list = [(0, 20, 0), (0, 45, 0), (0, 90, 0)]
            if light_mode:
                gaussians = self.gaussian_processor.get_model_object_from_mv(
                    images=images,
                    seed=seed_3d,
                    preprocess_image=True,
                    sparse_structure_sampler_params={"steps": 30, "cfg_strength": 7.5},
                    slat_sampler_params={"steps": 12, "cfg_strength": 3},
                    additional_euler_angles_list=rotate_list,
                    rotate=do_rotate,
                    prompt=prompt
                )
            else:
                gaussians = self.gaussian_processor.get_model_object_from_mv(
                    images=images,
                    seed=seed_3d,
                    preprocess_image=True,
                    sparse_structure_sampler_params={"steps": 30, "cfg_strength": 7.5},
                    slat_sampler_params={"steps": 25, "cfg_strength": 3},
                    additional_euler_angles_list=rotate_list,
                    rotate=do_rotate,
                    prompt=prompt
                )
            if not gaussians:
                raise RuntimeError("Gaussian processor failed to create a 3D model.")
            
            logger.info(f"Successfully generated 3D model with {len(gaussians)} candidates from the source image.")

        except Exception as e:
            logger.error(f"Error in I23D generation: {e}", exc_info=True)
            error_message = f"I23D Generation Error: {type(e).__name__}"
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        return gaussians, error_message



    def unload_models(self):
        logger.info("Unloading I23D models...")
        if self.gaussian_processor:
            self.gaussian_processor.unload_model()
        del self.gaussian_processor
        del self.rembg_model
        self.gaussian_processor = None
        self.rembg_model = None
        torch.cuda.empty_cache()
