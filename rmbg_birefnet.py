import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class BackgroundRemover:
    """
    Wrapper class for the RMBG-2.0 (BiRefNet) background removal model
    used in the Trellis 3D pipeline.
    """

    def __init__(
        self,
        model_name: str = "hiepnd11/rm_back2.0",
        image_size=(1024, 1024),
        device: str = "cuda"
    ):
        """
        Initialize the RMBG model.

        Args:
            model_name (str): Hugging Face model id for RMBG.
            image_size (tuple): Input size for inference.
            device (str): Device to use ("cuda" or "cpu").
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available. Falling back to CPU.")
            device = "cpu"
        self.device = device
        self.image_size = image_size

        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(device)
        self.model.eval()

        # Same normalization as Trellis preprocess
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def get_mask(self, image: Image.Image) -> np.ndarray:
        """
        Generate the object mask for an image (like _get_birefnet_mask).

        Args:
            image (PIL.Image): Input RGB image.

        Returns:
            np.ndarray: Binary mask (uint8, 0/1).
        """
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.autocast(
            device_type=self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32
        ):
            preds = self.model(input_tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()

        # Convert back to image and resize to match input
        mask_pil = transforms.ToPILImage()(pred)
        mask_resized = mask_pil.resize(image.size)
        mask_np = np.array(mask_resized)

        return (mask_np > 128).astype(np.uint8)

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply background removal and return RGBA image.

        Args:
            image (PIL.Image): Input RGB or RGBA image.

        Returns:
            PIL.Image: RGBA image with alpha from RMBG mask.
        """
        image = image.convert("RGB")
        mask = self.get_mask(image)
        rgba = image.convert("RGBA")
        rgba_np = np.array(rgba)
        rgba_np[:, :, 3] = mask * 255
        return Image.fromarray(rgba_np)
