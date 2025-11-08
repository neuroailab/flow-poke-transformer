import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from autoseg.utils import *

"""
Probability Motion Map Analysis + Point Sampling Vis

I wrote this scrit to to compute unconditional motion probability maps
and analyze how different magnitude thresholds affect point sampling
For each image, it:

1. Computes a motion probability heatmap using the Flow-Poke Transformer model, indicating 
   regions likely to move beyond a specified threshold (I set 12 pixels)
2. Samples diverse, spatially-separated points from high-probability motion regions using 
   multiple min_prob thresholds [0.001, 0.005, 0.01, 0.05, 0.1, 0.5] to understand how 
   selectivity affects the quantity and quality of sampled points
3. Generates visualizations & save results t h5

Output is used to determine optimal min_prob values for automatic segmentation initialization,
balancing point quantity (coverage) against motion confidence (quality).

   {image_key}.h5/
   ├── image                    # RGB image
   ├── prob_motion             # Probability heatmap (64x64)
   ├── expected_magnitude      # Expected magnitude map (64x64)
   ├── sampled_points/
   │   ├── min_prob_0.001      # Points for min_prob=0.001
   │   ├── min_prob_0.005      # Points for min_prob=0.005
   │   ├── ...
   │   └── min_prob_0.5        # Points for min_prob=0.5
   └── attrs: threshold, grid_size, min_dist, num_samples

"""

def visualize_prob_motion_and_samples(rgb, prob_motion, min_probs, min_dist, num_samples, output_path):
    n_min_probs = len(min_probs)
    n_cols = min(4, n_min_probs + 1)
    n_rows = 1 + ((n_min_probs + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Row 0, Col 0: Prob motion heatmap
    ax = axes[0, 0]
    im = ax.imshow(prob_motion, cmap='hot')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f'Prob Motion Heatmap\n(threshold=12 pixels)', fontsize=12, fontweight='bold')
    ax.axis('off')

    for col in range(1, n_cols):
        axes[0, col].axis('off')

    # sample pts + vis
    all_sampled_points = {}

    for idx, min_prob in enumerate(min_probs):
        # Sample points
        # sampled_pts = sample_diverse_high_prob_points(
        #     prob_motion, num_samples=num_samples, min_prob=min_prob, min_dist=min_dist
        # )
        sampled_pts, _, _, _ = sample_pmotion(rgb, num_samples, min_prob, min_dist, prob_motion, min_filter=True)
        sampled_pts = (np.array(sampled_pts) * 0.25).astype(int)
        all_sampled_points[min_prob] = sampled_pts

        row = 1 + (idx // n_cols)
        col = idx % n_cols

        if row >= n_rows:
            break

        ax = axes[row, col]

        ax.imshow(rgb)

        if len(sampled_pts) > 0: # scale to match
            H, W = rgb.shape[:2]
            prob_H, prob_W = prob_motion.shape
            sampled_pts_scaled = sampled_pts.copy()
            sampled_pts_scaled[:, 0] = sampled_pts[:, 0] * (W / prob_W)
            sampled_pts_scaled[:, 1] = sampled_pts[:, 1] * (H / prob_H)

            ax.scatter(sampled_pts_scaled[:, 0], sampled_pts_scaled[:, 1],
                       c='lime', s=50, marker='x', linewidths=2)

        ax.set_title(f'dist={min_dist}, min_prob={min_prob}\n{len(sampled_pts)} points sampled',
                     fontsize=12, fontweight='bold')
        ax.axis('off')

    for idx in range(n_min_probs, n_rows * n_cols - 1):
        row = 1 + (idx // n_cols)
        col = idx % n_cols
        if row < n_rows:
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return all_sampled_points


def main():
    # config
    dataset_path = "/ccn2/u/lilianch/data/550_openx_entity_dataset.h5"
    output_dir = "/ccn2/u/lilianch/external_repos/flow-poke-transformer/autosegp/pmotion12_minprob_samples_dist8_itersampling"

    h5_output_dir = os.path.join(output_dir, "h5")
    vis_output_dir = os.path.join(output_dir, "vis")
    os.makedirs(h5_output_dir, exist_ok=True)
    os.makedirs(vis_output_dir, exist_ok=True)

    # SET PARAMS (make sure to save these .. ..)
    threshold = 12
    min_probs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    min_dist = 8
    num_samples = 32
    grid_size = 64
    device = 'cuda'


    print("Loading FPT model")
    model = torch.hub.load(".", "fpt_base", source="local")
    model.eval()
    model.to(device)

    with h5py.File(dataset_path, 'r') as inp_data:
        all_keys = [k for k in inp_data.keys() if k.startswith('entityseg_') or k.startswith('openx_')]

        print(f"Found {len(all_keys)} examples in dataset")

        for img_key in tqdm(all_keys, desc="Processing images"):
            try:
                base_name = os.path.splitext(os.path.basename(img_key))[0]
                out_h5_path = os.path.join(h5_output_dir, f"{base_name}.h5")

                if os.path.exists(out_h5_path):
                    print(f"File {base_name} already exists, skipping")
                    continue

                rgb = inp_data[img_key]['rgb'][:]

                # motion maps
                print(f"Computing motion maps for {base_name}...")
                expected_mag_map, prob_motion_heatmap = compute_motion_maps(
                    Image.fromarray(rgb), model, grid_size=grid_size,
                    threshold=threshold, device=device
                )

                # visualize +_ sample pts
                print(f"Sampling points and creating visualization...")
                vis_path = os.path.join(vis_output_dir, f"{base_name}.jpg")
                all_sampled_points = visualize_prob_motion_and_samples(
                    rgb, prob_motion_heatmap, min_probs, min_dist, num_samples, vis_path
                )

                # save
                print(f"Saving to {out_h5_path}...")
                with h5py.File(out_h5_path, "w") as f:
                    f.create_dataset("image", data=rgb, compression="gzip")

                    f.create_dataset("prob_motion", data=prob_motion_heatmap, compression="gzip")
                    f.create_dataset("expected_magnitude", data=expected_mag_map, compression="gzip")

                    # assoc. metadata
                    f.attrs['threshold'] = threshold
                    f.attrs['grid_size'] = grid_size
                    f.attrs['min_dist'] = min_dist
                    f.attrs['num_samples'] = num_samples

                    # fore ach min_prob to test
                    sampled_points_group = f.create_group("sampled_points")
                    for min_prob, points in all_sampled_points.items():
                        if len(points) > 0:
                            sampled_points_group.create_dataset(
                                f"min_prob_{min_prob}", data=points, compression="gzip"
                            )
                        else:
                            # if no points, just emptuy
                            sampled_points_group.create_dataset(
                                f"min_prob_{min_prob}", data=np.empty((0, 2)), compression="gzip"
                            )


            except Exception as e:
                print(f"Error processing {img_key}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"H5 files saved to: {h5_output_dir}")
    print(f"Visualizations saved to: {vis_output_dir}")

if __name__ == "__main__":
    main()