import os
import sys
import numpy as np
from PIL import Image
import open3d as o3d
import requests
from gradio_client import Client, handle_file

def get_grounded_sam_mask(image_path, target_text):
    client = Client("mawady-uni/Grounded-SAM")
    result = client.predict(
        input_image=handle_file(image_path),
        text_prompt=target_text,
        task_type="seg",
        inpaint_prompt=target_text,
        box_threshold=0.3,
        text_threshold=0.25,
        iou_threshold=0.8,
        inpaint_mode="merge",
        api_name="/run_grounded_sam"
    )
    print(f"Grounded-SAM result: {result}")
    # result = [{'image': '/private/var/folders/1n/00nkn3s55gjcb8h2qkg9_mvc0000gn/T/gradio/f26a5bc4cae1e9117732a89dc1ed842b492fc4105496b3ddab56cf099aebd801/image.webp', 'caption': None}, {'image': '/private/var/folders/1n/00nkn3s55gjcb8h2qkg9_mvc0000gn/T/gradio/05039d4ea3d3924cb95d2b32fb04a8982ff2e93a70f3235fdd83906b9e7e2a87/image.webp', 'caption': None}]

    # Use only the first result
    mask_image_path = result[1]['image'] if isinstance(result[1]['image'], str) else result[1]['image']['path']
    # mask_image_path = "./image.webp"

    # # Load the mask (webp) as grayscale
    mask_img = Image.open(mask_image_path)
    if mask_img.mode != "L":
        mask_img = mask_img.convert("L")
    mask = np.array(mask_img) > 0  # Works for any nonzero mask

    print("Raw mask image min/max:", mask_img.getextrema())
    mask_array = np.array(mask_img)
    print("Unique mask values:", np.unique(mask_array))

    return mask

def main(example_path, target_text):
    api_token = os.environ.get("HF_TOKEN")
    if api_token is None:
        print("Please set your HuggingFace API token in the HF_TOKEN environment variable.")
        sys.exit(1)
    
    output_path = os.path.join(example_path, "output")
    images = sorted([f for f in os.listdir(os.path.join(example_path, "images")) if f.endswith(('.png', '.jpg', '.jpeg'))])
    for idx, img_name in enumerate(images):
        image_path = os.path.join(example_path, "images", img_name)
        depth_path = os.path.join(output_path, f"depth_map_{idx:03d}.npy")
        pc_path = os.path.join(output_path, f"pointcloud_{idx:03d}.npy")
        if not os.path.exists(depth_path) or not os.path.exists(pc_path):
            continue

        # 1. Get mask from Grounded-SAM
        mask = get_grounded_sam_mask(image_path, target_text)  # mask shape (H, W)

        # 2. Load depth and point cloud
        depth = np.load(depth_path)  # shape (H, W)
        points = np.load(pc_path)    # shape (H, W, 3) or (N, 3)
        if points.ndim == 2:
            # Already flattened
            print(f"Warning: Point cloud for {img_name} is already flat. Skipping.")
            continue

        # 3. Apply mask to points
        mask = mask.astype(bool)
        h_pts, w_pts = points.shape[:2]
        if mask.shape != (h_pts, w_pts):
            mask_resized = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((w_pts, h_pts), resample=Image.NEAREST)) > 127
        else:
            mask_resized = mask

        segmented_pts = points[mask_resized]

        # 4. (Optional) Get colors
        img_arr = np.array(Image.open(image_path)).astype(np.float32) / 255.0
        if img_arr.shape[:2] != mask_resized.shape:
            img_arr = np.array(Image.fromarray((img_arr * 255).astype(np.uint8)).resize(mask_resized.shape[::-1], resample=Image.BILINEAR)).astype(np.float32) / 255.0
        segmented_colors = img_arr[mask_resized]

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