import os
import sys
import numpy as np
from PIL import Image
import open3d as o3d
import requests

def get_grounded_sam_mask(image_path, target_text, api_token):
    # You need to use HuggingFace Inference API for Grounded-SAM
    api_url = "https://huggingface.co/spaces/mawady-uni/Grounded-SAM"
    headers = {"Authorization": f"Bearer {api_token}"}
    with open(image_path, "rb") as img_file:
        payload = {
            "inputs": {"image": img_file, "text": target_text}
        }
        response = requests.post(api_url, headers=headers, files={"image": img_file}, data={"text": target_text})
    response.raise_for_status()
    result = response.json()
    # This part may need to be adapted depending on output format from API
    mask = np.array(result["mask"], dtype=bool)
    return mask

def main(example_path, target_text):
    api_token = os.environ.get("HF_TOKEN")
    if api_token is None:
        print("Please set your HuggingFace API token in the HF_TOKEN environment variable.")
        sys.exit(1)
    
    print(f"hf token {api_token[:5]}")

    output_path = os.path.join(example_path, "output")
    images = sorted([f for f in os.listdir(os.path.join(example_path, "images")) if f.endswith(('.png', '.jpg', '.jpeg'))])
    for idx, img_name in enumerate(images):
        image_path = os.path.join(example_path, "images", img_name)
        depth_path = os.path.join(output_path, f"depth_map_{idx:03d}.npy")
        pc_path = os.path.join(output_path, f"pointcloud_{idx:03d}.npy")
        if not os.path.exists(depth_path) or not os.path.exists(pc_path):
            continue

        # 1. Get mask from Grounded-SAM
        mask = get_grounded_sam_mask(image_path, target_text, api_token)  # mask shape (H, W)

        # 2. Load depth and point cloud
        depth = np.load(depth_path)  # shape (H, W)
        points = np.load(pc_path)    # shape (H, W, 3) or (N, 3)
        if points.ndim == 2:
            # Already flattened
            print(f"Warning: Point cloud for {img_name} is already flat. Skipping.")
            continue

        # 3. Apply mask to points
        mask = mask.astype(bool)
        segmented_pts = points[mask]

        # 4. (Optional) Get colors
        img_arr = np.array(Image.open(image_path)).astype(np.float32) / 255.0
        if img_arr.shape[:2] != mask.shape:
            img_arr = np.array(Image.fromarray((img_arr * 255).astype(np.uint8)).resize(mask.shape[::-1])).astype(np.float32) / 255.0
        segmented_colors = img_arr[mask]

        # 5. Save segmented point cloud as .ply
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(segmented_pts)
        pc.colors = o3d.utility.Vector3dVector(segmented_colors)
        out_ply = os.path.join(output_path, f"segment_{target_text}_{idx:03d}.ply")
        o3d.io.write_point_cloud(out_ply, pc)
        print(f"Saved {out_ply}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python segment.py <example_folder> <target_text>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])