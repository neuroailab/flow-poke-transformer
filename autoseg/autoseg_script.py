import argparse
import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from motion_prob_exploration.utils import sample_pmotion, compute_motion_maps
from scripts.evaluate_spelkebench import get_segment, predict_flows, get_dot_product_map
from tqdm import tqdm


def visualize_detailed(rgb, gt_segment, prob_motion, sampled_pts,
                       flows, drags_pixel, segments, all_mean_dot_prod, num_dirs, output_path):
    fig, axes = plt.subplots(3, 4, figsize=(15, 20))

    # Row 1, Col 1: RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Image', fontweight='bold')
    axes[0, 0].axis('off')

    # Row 1, Col 2: GT Segment
    axes[0, 1].imshow(gt_segment[0], cmap='gray')
    axes[0, 1].set_title('GT Segment', fontweight='bold')
    axes[0, 1].axis('off')

    # Row 1, Col 3: Prob motion + sampled points
    # axes[0, 2].imshow(rgb)
    H, W = rgb.shape[:2]
    prob_motion_resized = cv2.resize(prob_motion, (H,W))
    axes[0, 2].imshow(prob_motion_resized, cmap='hot', alpha=0.5)
    axes[0, 2].scatter(sampled_pts[:, 0], sampled_pts[:, 1],
                       c='lime', s=50, marker='x', linewidths=2)
    axes[0, 2].set_title(f'Prob Motion + Samples ({len(sampled_pts)} pts)', fontweight='bold')
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')

    # Rows 2-4: Sample points 0 and 9 (indices 0 and 9, showing first direction)
    sample_indices = [0, 9] if len(sampled_pts) > 9 else [0, min(9, len(sampled_pts) - 1)]

    for row_idx, sample_idx in enumerate(sample_indices):
        if sample_idx >= len(sampled_pts):
            continue

        flow_idx = sample_idx * num_dirs  # First direction for this sample

        if flow_idx >= len(flows):
            continue

        flow = flows[flow_idx]  # [H, W, 2]
        drag = drags_pixel[flow_idx]  # (x1, y1, x2, y2)
        drag_scale = 256/448
        segment = segments[sample_idx]
        mean_dot_prod = all_mean_dot_prod[sample_idx]

        row = row_idx + 1

        # Col 1: Flow visualization + poke arrow
        from torchvision.utils import flow_to_image
        from einops import rearrange
        flow_img = rearrange(flow_to_image(rearrange(flow, "h w c -> c h w").float().cpu()),
                             "c h w -> h w c").numpy()
        axes[row, 0].imshow(flow_img)
        # Draw poke arrow
        axes[row, 0].arrow(drag[0]*drag_scale, drag[1]*drag_scale, (drag[2]*drag_scale) - (drag[0]*drag_scale), (drag[3]*drag_scale) - (drag[1]*drag_scale),
                           color='green', width=3, head_width=10, length_includes_head=True)
        axes[row, 0].set_title(f'Sample {sample_idx + 1}: Flow + Poke', fontweight='bold')
        axes[row, 0].axis('off')

        # Col 2: Dot product map
        dot_prod_map = get_dot_product_map(flow, drag)
        im = axes[row, 1].imshow(dot_prod_map.cpu().numpy(), cmap='RdBu_r')
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046)
        axes[row, 1].set_title(f'Sample {sample_idx + 1}: Dot Product for 1 flow', fontweight='bold')
        axes[row, 1].axis('off')

        # Col 2: Mean Dot product map
        im = axes[row, 2].imshow(mean_dot_prod, cmap='RdBu_r')
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)
        axes[row, 2].set_title(f'Sample {sample_idx + 1}: Mean dot product for 5 flows', fontweight='bold')
        axes[row, 2].axis('off')

        # Col 4: Predicted segment
        axes[row, 3].imshow(rgb)
        axes[row, 3].imshow(segment, cmap='jet', alpha=0.5)
        axes[row, 3].scatter([sampled_pts[sample_idx, 0]], [sampled_pts[sample_idx, 1]],
                             c='lime', s=100, marker='x', linewidths=3)
        axes[row, 3].set_title(f'Sample {sample_idx + 1}: Pred Segment (thresholded mean dot prod)', fontweight='bold')
        axes[row, 3].axis('off')

    # for row in range(len(sample_indices) + 1, 4):
    #     for col in range(2):
    #         axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_all_segments(rgb, sampled_pts, segments, output_path):
    """
    8x4 grid showing all predicted segments overlaid on RGB with their sample points
    """
    n_samples = len(segments)
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(n_samples):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # rgb + seg overaly
        ax.imshow(rgb)
        ax.imshow(segments[idx], cmap='jet', alpha=0.5)
        ax.scatter([sampled_pts[idx, 0]], [sampled_pts[idx, 1]],
                   c='lime', s=100, marker='x', linewidths=3)
        ax.set_title(f'Segment {idx + 1}\nPoint: ({sampled_pts[idx, 0]:.0f}, {sampled_pts[idx, 1]:.0f})',
                     fontweight='bold')
        ax.axis('off')

    for idx in range(n_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def run_autoseg(model, dataset_path, output_dir, num_dirs, min_mag, max_mag,
                num_samples, start_idx=None, end_idx=None, gpu_id=None):
    # output directories
    h5_dir = os.path.join(output_dir, "h5")
    vis_dir = os.path.join(output_dir, "vis")
    vis2_dir = os.path.join(output_dir, "seg_vis")
    os.makedirs(h5_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(vis2_dir, exist_ok=True)

    # use p motino from the explore_minprob script in this fodler
    p_motion_dir = '/ccn2/u/lilianch/external_repos/flow-poke-transformer/autosegp/pmotion12_minprob_samples/h5'

    gpu_prefix = f"[GPU {gpu_id}] " if gpu_id is not None else ""

    with h5py.File(dataset_path, 'r') as inp_data:
        all_keys = [k for k in inp_data.keys() if k.startswith('entityseg_') or k.startswith('openx_')]

        # slicing (for paralleel)
        if start_idx is not None and end_idx is not None:
            all_keys = all_keys[start_idx:end_idx]
            print(f"{gpu_prefix}Processing examples {start_idx}-{end_idx} ({len(all_keys)} examples)")

        for example_idx, img_path in enumerate(tqdm(all_keys, desc=f"{gpu_prefix}Processing")):
            try:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                out_h5_path = os.path.join(h5_dir, f"{base_name}.h5")
                p_motion_path = os.path.join(p_motion_dir, f"{base_name}.h5")
                p_motion_file = h5py.File(p_motion_path, 'r')

                if os.path.exists(out_h5_path):
                    print(f"{gpu_prefix}File {base_name} already exists, skipping...")
                    continue

                rgb = inp_data[img_path]['rgb'][:]
                gt_segments = inp_data[img_path]['segment'][:]

                # Compute motion map once per image
                # print(f"{gpu_prefix}Computing motion maps for {base_name}...")
                # expected_mag_map, prob_motion = compute_motion_maps(
                #     Image.fromarray(rgb), model, grid_size=64,
                #     threshold=min_mag, device='cuda'
                # )
                # breakpoint()
                prob_motion = p_motion_file['prob_motion'][:]

                # Sample points from motion map
                # print(f"{gpu_prefix}Sampling points...")
                # sampled_pts, prob_maps, _, _ = sample_pmotion(
                #     rgb, num_samples, min_prob, min_dist, prob_motion, min_filter=min_filter
                # )
                # breakpoint()
                sampled_pts = p_motion_file['sampled_points']['min_prob_0.001'][:]
                # sampled_pts = sampled_pts[:num_samples] * (256/64)
                sampled_pts = sampled_pts * (256/64)

                centroids = torch.tensor(sampled_pts, dtype=torch.float32)

                # generate segments
                print(f"{gpu_prefix}Generating segments for {len(centroids)} points...")
                all_pred_segs = []
                all_mean_dot_prod = []

                # 448 for model
                target_size = 448
                H, W = rgb.shape[:2]
                rgb_resized = np.array(Image.fromarray(rgb).resize((target_size, target_size), Image.BICUBIC))

                # normalize to [0,1]
                centroids_resized = centroids.clone()
                centroids_resized[:, 0] = centroids[:, 0] * (target_size / W)
                centroids_resized[:, 1] = centroids[:, 1] * (target_size / H)
                centroids_norm = centroids_resized / target_size

                # Get flows for all centroids
                flows, drags_pixel = predict_flows(
                    model=model,
                    rgb=rgb_resized,
                    centroids_norm=centroids_norm.to('cuda'),
                    num_offsets=num_dirs,
                    flow_resolution=256,
                    prediction_mode='parallel',
                    min_mag=min_mag / 256.0,
                    max_mag=max_mag / 256.0,
                    device=torch.device('cuda'),
                    query_batch_size=4096,
                    target_size=target_size
                )

                # generate segs from flow (using dot prod method)
                from scripts.evaluate_spelkebench import segment_from_flows
                for i in range(len(centroids)):
                    flow_start = i * num_dirs
                    flow_end = flow_start + num_dirs
                    segment, mean_dot_prod_map = segment_from_flows(
                        flows[flow_start:flow_end],
                        drags_pixel[flow_start:flow_end],
                        num_dirs
                    )
                    all_pred_segs.append(segment)
                    all_mean_dot_prod.append(mean_dot_prod_map)

                all_pred_segs = np.stack(all_pred_segs)

                # Save to H5
                print(f"{gpu_prefix}Saving to {out_h5_path}...")
                with h5py.File(out_h5_path, "w") as f:
                    f.create_dataset("image", data=rgb, compression="gzip")
                    f.create_dataset("segment_gt", data=gt_segments, compression="gzip")
                    f.create_dataset("segment_pred", data=all_pred_segs, compression="gzip")
                    f.create_dataset("sampled_points", data=sampled_pts, compression="gzip")
                    f.create_dataset("prob_motion", data=prob_motion, compression="gzip")
                    # f.create_dataset("expected_magnitude", data=expected_mag_map, compression="gzip")

                # Visualizations
                print(f"{gpu_prefix}Creating visualizations...")

                # Vis 1: Detailed view
                vis1_path = os.path.join(vis_dir, f"{base_name}_detailed.jpg")
                visualize_detailed(rgb, gt_segments, prob_motion, sampled_pts,
                                   flows, drags_pixel, all_pred_segs, all_mean_dot_prod, num_dirs, vis1_path)

                # def visualize_detailed(rgb, gt_segment, prob_motion, sampled_pts,
                #                        flows, drags_pixel, segments, mean_dot_prod_map, num_dirs, output_path):

                # Vis 2: all segments (only for first 10 examples in this chunk)
                if example_idx < 30:
                    vis2_path = os.path.join(vis2_dir, f"{base_name}_all_segments.jpg")
                    visualize_all_segments(rgb, sampled_pts, all_pred_segs, vis2_path)

                print(f"{gpu_prefix}✓ Completed {base_name}")

            except Exception as e:
                print(f"{gpu_prefix}✗ Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                continue


def main(args):
    print("Loading FPT model")
    model = torch.hub.load(".", "fpt_base", source="local")
    model.eval()
    model.to('cuda')

    run_autoseg(
        model=model,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_dirs=args.num_dirs,
        min_mag=args.min_mag,
        max_mag=args.max_mag,
        num_samples=args.num_samples,
        start_idx=None,
        end_idx=None,
        gpu_id=None
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='/ccn2/u/lilianch/external_repos/flow-poke-transformer/550_openx_entity_dataset.h5')
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_dirs", type=int, default=5, help="Number of poke directions per sample point")
    parser.add_argument("--min_mag", type=float, default=10.0, help="Min poke magnitude in pixels")
    parser.add_argument("--max_mag", type=float, default=25.0, help="Max poke magnitude in pixels")
    parser.add_argument("--num_samples", type=int, default=32, help="Target number of sample points")

    args = parser.parse_args()

    main(args)

