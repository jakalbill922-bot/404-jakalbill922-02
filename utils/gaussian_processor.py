import gc
import time
from io import BytesIO
from rotate3d import rotate_3d
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple

from trellis.pipelines import TrellisImageTo3DPipeline

from trellis.representations.gaussian import Gaussian

from scipy.spatial.transform import Rotation

import logging

logger = logging.getLogger(__name__)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class GaussianProcessor:
    """Generates 3d models and videos"""

    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._image_to_3d_pipeline: TrellisImageTo3DPipeline | None = None

    def preload_model(self, model_name: str = "Stable-X/trellis-vggt-v0-2"):
        self._image_to_3d_pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
        self._image_to_3d_pipeline.to(self._device)
        torch.cuda.empty_cache()

    def unload_model(self):
        del self._image_to_3d_pipeline
        self._image_to_3d_pipeline = None

        torch.cuda.empty_cache()
        gc.collect()

    # async def get_model_ply(self, image:Image, seed:int, preprocess_image:bool = True, sparse_structure_sampler_params: dict = {}, slat_sampler_params: dict = {}) -> BytesIO:
    #     """
    #     Generates a 3D model in PLY format

    #     Args:
    #         image: a PIL image
    #         prompt (str): The prompt from which to generate a 3D model
    #         seed (int): random seed for trellis model generator
    #         preprocess_image (bool): enables / disables image preprocessing
    #     Returns:
    #         BytesIO: The Buffer object containing the 3D model in PLY format
    #     """

    #     buffer = BytesIO()
    #     outputs = self._image_to_3d_pipeline.run(
    #         image,
    #         seed=seed,
    #         preprocess_image=preprocess_image,
    #         sparse_structure_sampler_params=sparse_structure_sampler_params,
    #         slat_sampler_params=slat_sampler_params
    #     )

    #     gaussians: Gaussian = outputs["gaussian"][0]
    #     T = np.array([0, 0, 0])
    #     R_initial_obj = Rotation.from_euler('xyz', [90.0, 0.0, 0.0], degrees=True)
    #     R = R_initial_obj.as_matrix().astype(np.float32)
    #     gaussians.transform_data(T, R)
    #     gaussians.save_ply(buffer)
    #     buffer.seek(0)

    #     return buffer

    def get_model_object(
        self,
        images: List[Image.Image],
        seed: int,
        preprocess_image: bool = True,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        additional_euler_angles_list: List[Union[List[float], Tuple[float, float, float]]] = None,
        rotate: bool = True,
        prompt: str = "",
        task_type: str = "text"
    ) -> List[Gaussian]:
        """
        Generates multiple 3D Gaussian objects with additional rotations applied.

        Args:
            images: a list of PIL images
            seed (int): random seed for trellis model generator
            preprocess_image (bool): enables / disables image preprocessing
            sparse_structure_sampler_params (dict): parameters for sparse structure sampler
            slat_sampler_params (dict): parameters for slat sampler
            additional_euler_angles_list (List[List[float]]): A list of additional Euler angles
                (each as [rx, ry, rz] in degrees, xyz order) to apply after the initial rotation.
                If None or empty, returns a list containing only the initially rotated model.
        Returns:
            List[Gaussian]: A list of Gaussian objects, each corresponding to an
                            additional rotation applied to the initially rotated model.
        """
        t0 = time.time()
        if self._image_to_3d_pipeline is None:
            raise RuntimeError("Model not preloaded. Call preload_model() first.")

        print("GaussianProcessor: Running image to 3D pipeline...")
        images = [self._image_to_3d_pipeline.preprocess_image(image) for image in images]
        ########
        torch.cuda.empty_cache()
        gc.collect() 
        ########
        outputs = self._image_to_3d_pipeline.run_multi_image(
            images,
            seed=seed,
            num_samples=1,
            preprocess_image=False,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            slat_sampler_params=slat_sampler_params,
            mode="multidiffusion"
        )

        if not outputs["gaussian"]:
            raise ValueError("Pipeline did not return any Gaussian objects.")

        logger.info(f'GaussianProcessor: Received base Gaussian object with {len(outputs["gaussian"])} gaussion')

        # --- Lấy và xoay Gaussian gốc ---
        base_gaussians: Gaussian = outputs["gaussian"][0]
        logger.info(f"GaussianProcessor: Received base Gaussian object with {base_gaussians._xyz.shape[0]} points.")

        T_initial = np.array([0, 0, 0], dtype=np.float32)
        R_initial_obj = Rotation.from_euler('xyz', [-90.0, 0.0, 0.0], degrees=True)
        R_initial = R_initial_obj.as_matrix().astype(np.float32)

        logger.info("GaussianProcessor: Applying initial transform...")
        base_gaussians.transform_data(T_initial, R_initial)
        # base_gaussians bây giờ đã ở trạng thái xoay ban đầu

        if rotate and task_type == "text":
            width, height, depth = base_gaussians.get_dims()
            angles = rotate_3d(width, height, depth, prompt)
            if angles is not None:
                T_initial = np.array([0, 0, 0], dtype=np.float32)
                R_initial_obj = Rotation.from_euler('xyz', angles, degrees=True)
                R_initial = R_initial_obj.as_matrix().astype(np.float32)
                base_gaussians.transform_data(T_initial, R_initial)

        results_list: List[Gaussian] = []
        T_additional = np.array([0, 0, 0], dtype=np.float32)

        results_list.append(base_gaussians.copy())
        
        if additional_euler_angles_list:
            # logger.info(f"GaussianProcessor: Applying {len(additional_euler_angles_list)} additional rotations...")
            for i, euler_angles in enumerate(additional_euler_angles_list):
                # logger.info(f"  Applying rotation {i+1}/{len(additional_euler_angles_list)}: {euler_angles}")
                # --- Tạo bản sao và áp dụng xoay bổ sung ---
                gaussians_copy = base_gaussians.copy() # Tạo bản sao từ trạng thái đã xoay ban đầu

                R_additional_obj = Rotation.from_euler('xyz', euler_angles, degrees=True)
                R_additional = R_additional_obj.as_matrix().astype(np.float32)

                gaussians_copy.transform_data(T_additional, R_additional)
                # ------------------------------------------

                # --- Thêm bản sao đã xoay vào danh sách kết quả ---
                results_list.append(gaussians_copy)
                # ----------------------------------------------

        logger.info(f"GaussianProcessor: Finished processing. Returning {len(results_list)} Gaussian object(s).")
        t1 = time.time()
        print(f"[GEN3D TIME] Time taken: {t1 - t0:0.2f} seconds")
        return results_list

    def get_model_object_from_mv(
        self,
        images: List[Image.Image],
        seed: int,
        preprocess_image: bool = True,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        additional_euler_angles_list: List[Union[List[float], Tuple[float, float, float]]] = None,
        rotate: bool = True
    ) -> List[Gaussian]:
        """
        Generates multiple 3D Gaussian objects with additional rotations applied.

        Args:
            image: a PIL image
            seed (int): random seed for trellis model generator
            preprocess_image (bool): enables / disables image preprocessing
            sparse_structure_sampler_params (dict): parameters for sparse structure sampler
            slat_sampler_params (dict): parameters for slat sampler
            additional_euler_angles_list (List[List[float]]): A list of additional Euler angles
                (each as [rx, ry, rz] in degrees, xyz order) to apply after the initial rotation.
                If None or empty, returns a list containing only the initially rotated model.
        Returns:
            List[Gaussian]: A list of Gaussian objects, each corresponding to an
                            additional rotation applied to the initially rotated model.
        """
        if self._image_to_3d_pipeline is None:
            raise RuntimeError("Model not preloaded. Call preload_model() first.")

        print("GaussianProcessor: Running image to 3D pipeline...", len(images))
        images = [self._image_to_3d_pipeline.preprocess_image(image) for image in images]
        # normalized_images = [self.normal_predictor(image, resolution=512, match_input_resolution=True, data_type='object') for image in images]
        ########
        torch.cuda.empty_cache()
        gc.collect() 
        ########
        outputs = self._image_to_3d_pipeline.run_multi_image(
            images,
            seed=seed,
            num_samples=1,
            preprocess_image=False,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            slat_sampler_params=slat_sampler_params,
            mode="multidiffusion"
        )

        if not outputs["gaussian"]:
            raise ValueError("Pipeline did not return any Gaussian objects.")

        logger.info(f'GaussianProcessor: Received base Gaussian object with {len(outputs["gaussian"])} gaussion')

        # --- Lấy và xoay Gaussian gốc ---
        base_gaussians: Gaussian = outputs["gaussian"][0]
        logger.info(f"GaussianProcessor: Received base Gaussian object with {base_gaussians._xyz.shape[0]} points.")

        T_initial = np.array([0, 0, 0], dtype=np.float32)
        R_initial_obj = Rotation.from_euler('xyz', [-90.0, 0.0, 0.0], degrees=True)
        R_initial = R_initial_obj.as_matrix().astype(np.float32)

        logger.info("GaussianProcessor: Applying initial transform...")
        base_gaussians.transform_data(T_initial, R_initial)
        # base_gaussians bây giờ đã ở trạng thái xoay ban đầu

        if rotate:
            width, height, depth = base_gaussians.get_dims()
            angles = rotate_3d(width, height, depth, prompt)
            if angles is not None:
                T_initial = np.array([0, 0, 0], dtype=np.float32)
                R_initial_obj = Rotation.from_euler('xyz', angles, degrees=True)
                R_initial = R_initial_obj.as_matrix().astype(np.float32)
                base_gaussians.transform_data(T_initial, R_initial)

        results_list: List[Gaussian] = []
        T_additional = np.array([0, 0, 0], dtype=np.float32)

        results_list.append(base_gaussians.copy())
        
        if additional_euler_angles_list and rotate:
            # logger.info(f"GaussianProcessor: Applying {len(additional_euler_angles_list)} additional rotations...")
            for i, euler_angles in enumerate(additional_euler_angles_list):
                # logger.info(f"  Applying rotation {i+1}/{len(additional_euler_angles_list)}: {euler_angles}")
                # --- Tạo bản sao và áp dụng xoay bổ sung ---
                gaussians_copy = base_gaussians.copy() # Tạo bản sao từ trạng thái đã xoay ban đầu

                R_additional_obj = Rotation.from_euler('xyz', euler_angles, degrees=True)
                R_additional = R_additional_obj.as_matrix().astype(np.float32)

                gaussians_copy.transform_data(T_additional, R_additional)
                # ------------------------------------------

                # --- Thêm bản sao đã xoay vào danh sách kết quả ---
                results_list.append(gaussians_copy)
                # ----------------------------------------------

        logger.info(f"GaussianProcessor: Finished processing. Returning {len(results_list)} Gaussian object(s).")
        return results_list

    def rotate_and_flip_gaussian(self, gaussian, euler_angles_deg):
        # Tạo bản sao
        gaussian_rotated = gaussian.copy()
        
        # === BƯỚC 1: LẬT HỆ TỌA ĐỘ (FLIP) ===
        # Ma trận lật hệ tọa độ Y-down, Z-in
        M = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        T_flip = np.array([0, 0, 0], dtype=np.float32)
        # Áp dụng phép lật. Sau bước này, gaussian tạm thời ở trong không gian "lạ"
        gaussian_rotated.transform_data(T_flip, M)

        # === BƯỚC 2: XOAY TRONG HỆ TỌA ĐỘ ĐÃ LẬT ===
        # Góc xoay (dựa trên thực nghiệm, góc dương là đúng)
        angle_y_deg = euler_angles_deg[1]
        # Ma trận xoay đơn giản quanh trục Y
        R_cam = Rotation.from_euler('y', angle_y_deg, degrees=True).as_matrix().astype(np.float32)
        T_rotate = np.array([0, 0, 0], dtype=np.float32)
        # Áp dụng phép xoay
        gaussian_rotated.transform_data(T_rotate, R_cam)

        # === BƯỚC 3: LẬT NGƯỢC HỆ TỌA ĐỘ VỀ BAN ĐẦU ===
        # Ma trận lật ngược (M_inv) chính là M
        M_inv = M
        T_flip_back = np.array([0, 0, 0], dtype=np.float32)
        # Áp dụng phép lật ngược
        gaussian_rotated.transform_data(T_flip_back, M_inv)

        return gaussian_rotated