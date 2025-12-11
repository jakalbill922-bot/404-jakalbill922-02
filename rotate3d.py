
from PIL import Image
import os
import json

def rotate_3d(width, height, depth, prompt):
    try:
        dims = [depth, height, width]
        min_dim, median_dim, max_dim = sorted(dims)

        min_dim_name = "depth" if min_dim == depth else "width" if min_dim == width else "height"
        median_dim_name = "depth" if median_dim == depth else "width" if median_dim == width else "height"
        max_dim_name = "depth" if max_dim == depth else "width" if max_dim == width else "height"

        with open("rotate_dict.json", "r") as f:
            rotate_dict = json.load(f)
            rotate_3d = rotate_dict.get("rotate_3d", [])

        for rotate_obj in rotate_3d:
            rotate_words = rotate_obj.get("rotate_words", [])
            obj_min_dim_name = rotate_obj.get("min_dim_name", "")
            obj_max_dim_name = rotate_obj.get("max_dim_name", "")
            angles = rotate_obj.get("angle", [-45, 0, 45])
            if any(word in prompt.lower().split() for word in rotate_words):
                cmp_dim_name = ""

                if obj_min_dim_name and min_dim_name in obj_min_dim_name:
                    limit_min_dim = rotate_obj.get("limit_min_dim", 0.3)
                    if min_dim < limit_min_dim:
                        cmp_dim_name = max_dim_name

                if obj_max_dim_name and max_dim_name in obj_max_dim_name:
                    cmp_dim_name = min_dim_name

                if cmp_dim_name == "width":
                    return [0, 0, angles[2]]
                elif cmp_dim_name == "depth":
                    return [angles[0], 0, 0]
                else:
                    return [0, angles[1], 0]


    except Exception as e:
        print(f"[ROTATE 3D] Error: {e}")

    print("[ROTATE 3D] No rotation needed")
    return None
