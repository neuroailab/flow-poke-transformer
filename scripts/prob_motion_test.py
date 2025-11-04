import h5py as h5
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os
from pathlib import Path

# Load FPT model
print("Loading FPT model...")
model = torch.hub.load(".", "fpt_base", source="local")
model.eval()
model.to('cuda')
print("Model loaded!")

# Configuration
gt_path = '/ccn2/u/lilianch/data/550_openx_entity_dataset.h5'
output_dir = '/ccn2/u/lilianch/external_repos/flow-poke-transformer/motion_prob_exploration/vis'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

grid_size = 64
device = 'cuda'
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]


def process_image_for_model(image):
    # format: (1, 3, H, W) in [-1, 1]
    image = image.resize((256, 256), Image.BICUBIC)
    image_np = np.array(image).astype(np.float32) / 255.0  # [0, 1]
    image_t = torch.from_numpy(image_np).permute(2, 0, 1)  # (3, H, W)
    image_t = image_t.unsqueeze(0)  # (1, 3, H, W)
    image_t = image_t * 2 - 1  # [-1, 1]
    return image_t.to('cuda')


def compute_motion_maps(image, model, grid_size, thresholds, device):
    from flow_poke.model import make_axial_pos_2d

    image_t = process_image_for_model(image)

    # Create coordinate grid ; i.e. [0, 1] divided into 64 bins w/ center as point
    # 2/64 - 1/64 = 0.0078125, first point is (0.0078125, 0.0078125) for coords[0]
    coords = make_axial_pos_2d(grid_size, grid_size, device=device)  # (4096, 2), same as 64x64x2
    coords_grid = coords.reshape(grid_size, grid_size, 2)  # (64, 64, 2)


    # Get unconditional predictions (no poke)
    d_img = model.embed_image(image_t)

    poke_pos_empty = torch.empty((1, 0, 2), dtype=torch.float32, device=device)
    poke_flow_empty = torch.empty((1, 0, 2), dtype=torch.float32, device=device)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        pred = model.predict_parallel(
            poke_pos=poke_pos_empty,
            poke_flow=poke_flow_empty,
            query_pos=coords[None],  # (1, 4096, 2)
            camera_static=True,
            d_img=d_img
        )

    pred_single = pred[0]  # Remove batch dimension, torch.Size([4096, 4])
    dense_prob_maps = torch.zeros(4096, grid_size, grid_size, device=device)

    for i in tqdm(range(4096), desc="Computing dense maps"):
        query_pos = coords[i]  # (2,)

        # flow vectors from query_i to all grid positions
        flows_to_all = coords_grid - query_pos[None, None, :]  # (64, 64, 2)
        flows_flat = flows_to_all.reshape(-1, 2)  # (4096, 2)

        # gmm for query
        gmm_i = pred_single[i]

        # evaluate GMM probability for each possible flow
        with torch.no_grad():
            log_probs = gmm_i.log_prob(flows_flat)  # (4096,)
            probs = torch.exp(log_probs)

        # Reshape to spatial map
        dense_prob_maps[i] = probs.reshape(grid_size, grid_size)

    # reshsape (64x64)
    prob_motion_heatmaps = {}
    expected_magnitude_map = torch.zeros(4096, device=device)
    threshold_first = thresholds[0]
    for threshold in thresholds:
        prob_motion_map = torch.zeros(4096, device=device)

        for i in tqdm(range(4096), desc="Aggregating"):
            query_pos = coords[i]

            # flow/mag
            flows_to_all = coords_grid - query_pos[None, None, :]  # (64, 64, 2)
            flow_mags = torch.norm(flows_to_all, dim=-1)  # (64, 64)
            motion_mask = (flow_mags > threshold).float()  # (64, 64)

            # p(motion) = sum of probabilities (above mag thresh)
            prob_motion_map[i] = (dense_prob_maps[i] * motion_mask).sum()

            # exp mag = weighted average of magnitudes
            if threshold == threshold_first:
                expected_magnitude_map[i] = (dense_prob_maps[i] * flow_mags).sum()

        prob_motion_heatmaps[threshold] = prob_motion_map.reshape(grid_size, grid_size).cpu().numpy()
    expected_magnitude_map = expected_magnitude_map.reshape(grid_size, grid_size).cpu().numpy()

    return expected_magnitude_map, prob_motion_heatmaps


def visualize_motion_analysis(image, expected_mag_heatmap, prob_motion_heatmaps,
                              thresholds, output_path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 0, Col 0: og img
    ax = axes[0, 0]
    ax.imshow(image)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Row 0, Col 1: exp mag
    ax = axes[0, 1]
    im = ax.imshow(expected_mag_heatmap, cmap='viridis')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title('Expected Flow Magnitude\nE[||flow||]', fontsize=12, fontweight='bold')
    ax.axis('off')

    # rest: thresholds
    positions = [(0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]

    for idx, threshold in enumerate(thresholds):
        row, col = positions[idx]
        ax = axes[row, col]

        prob_map = prob_motion_heatmaps[threshold]
        im = ax.imshow(prob_map, cmap='hot', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f'P(motion > {threshold})\nMean: {prob_map.mean():.3f}',
                     fontsize=12, fontweight='bold')
        ax.axis('off')

    axes[1, 3].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# h5 file process
print(f"\nOpening h5 file: {gt_path}")
gt_file = h5.File(gt_path, 'r')
all_keys = list(gt_file.keys())
print(f"Found {len(all_keys)} examples in h5 file")

# process
for idx, example_key in enumerate(tqdm(all_keys, desc="Processing examples")):
    try:
        rgb = gt_file[example_key]['rgb'][:]
        image = Image.fromarray(rgb)

        expected_mag_heatmap, prob_motion_heatmaps = compute_motion_maps(
            image, model, grid_size, thresholds, device
        )

        output_filename = f"{example_key}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        visualize_motion_analysis(
            image,
            expected_mag_heatmap,
            prob_motion_heatmaps,
            thresholds,
            output_path
        )

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(all_keys)} examples")

    except Exception as e:
        print(f"Error processing {example_key}: {e}")
        continue

# Clean up
gt_file.close()
print(f"Saved {len(all_keys)} visualizations to: {output_dir}")