import os
import torch
import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt

def save_depth_vis(depth, out_path):
    vmin = np.nanmin(depth)
    vmax = np.nanmax(depth)
    depth_vis = (depth - vmin) / (vmax - vmin + 1e-8)
    plt.imsave(out_path, depth_vis, cmap='turbo')

def run_and_save(target_dir, output_subdir="output"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model (adapt to your setup if needed)
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)

    # Gather images
    image_dir = os.path.join(target_dir, "images")
    image_names = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))])
    if not image_names:
        raise ValueError("No images found in images/ subfolder!")

    # Inference
    images = load_and_preprocess_images(image_names).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)

    # Pose/intrinsic conversion
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # To numpy
    for k in predictions.keys():
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = predictions[k].cpu().numpy().squeeze(0)

    # World points
    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # --- Saving section ---
    output_dir = os.path.join(target_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Camera poses
    for i, extr in enumerate(predictions['extrinsic']):
        np.save(os.path.join(output_dir, f'camera_pose_{i:03d}.npy'), extr)
    # Depth maps
    for i, depth in enumerate(predictions['depth']):
        np.save(os.path.join(output_dir, f'depth_map_{i:03d}.npy'), depth[..., 0])
        # Save colorized depth image
        depth_img_path = os.path.join(output_dir, f"depth_map_{i:03d}.png")
        save_depth_vis(depth[..., 0], depth_img_path)
    # Point clouds
    for i, pts in enumerate(predictions['world_points_from_depth']):
        np.save(os.path.join(output_dir, f'pointcloud_{i:03d}.npy'), pts)
        rgb_img = np.array(Image.open(image_names[i])).astype(np.float32) / 255.0  # (H_img, W_img, 3)
        # Resize if necessary
        if rgb_img.shape[:2] != pts.shape[:2]:
            rgb_img = np.array(Image.fromarray((rgb_img * 255).astype(np.uint8)).resize((pts.shape[1], pts.shape[0]), resample=Image.BILINEAR)).astype(np.float32) / 255.0

        valid_depth = np.isfinite(pts).all(axis=2)  # (H, W)
        pts_flat = pts[valid_depth]      # (N, 3)
        colors_flat = rgb_img[valid_depth]  # (N, 3)

        print(f"pts_flat: {pts_flat.shape}, colors_flat: {colors_flat.shape}")

        if pts_flat.size == 0 or pts_flat.shape[1] != 3:
            print(f"Skipping empty or invalid pointcloud for frame {i}: shape {pts.shape}")
            continue
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts_flat)
        pc.colors = o3d.utility.Vector3dVector(colors_flat)
        ply_name = f'pointcloud_{i:03d}.ply'
        ply_path = os.path.join(output_dir, ply_name)
        o3d.io.write_point_cloud(ply_path, pc)

    print(f"All outputs saved to {output_dir}")

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python demo_extract.py <target_dir>")
#         exit(1)
#     run_and_save(sys.argv[1])

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python demo_custom.py <parent_dir>")
        exit(1)
    parent_dir = sys.argv[1]
    for folder in sorted(os.listdir(parent_dir)):
        full_path = os.path.join(parent_dir, folder)
        if os.path.isdir(full_path):
            print(f"\nProcessing {full_path}")
            try:
                run_and_save(full_path)
            except Exception as e:
                print(f"Failed to process {full_path}: {e}")