# Generate flows from P motion sampled points w / FPT
# Saves probe_points and flows.

import os
import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from tqdm import tqdm
from torch.distributions import MixtureSameFamily
import matplotlib.pyplot as plt
from torchvision.utils import flow_to_image
from flow_poke.model import FlowPokeTransformer_Base, make_axial_pos_2d
from scripts.utils import offset_multiple_centroids, normalized_to_pixel


def predict_flows_poke_pos(
        model,
        rgb: np.ndarray,
        centroids_norm: torch.Tensor,
        num_offsets: int = 15,
        flow_resolution: int = 256,
        min_mag: float = 10.0 / 256.0,
        max_mag: float = 25.0 / 256.0,
        device: torch.device = None,
        query_batch_size: int = 4096,
        target_size: int = 448
):
    """
    Generate flows for multiple centroids w/ multiple offset directions.
    Returns flows: (N*num_offsets, H, W, 2) tensor &  probe_points_pixel: (N, num_offsets, 2, 2) array - (start, end) in pixel coords
    """
    N = centroids_norm.shape[0]

    # query pos + offsets
    query_pos = make_axial_pos_2d(flow_resolution, flow_resolution, device=device)[None]
    dx, dy = offset_multiple_centroids(centroids_norm, num_offsets, min_mag, max_mag)

    # image embd
    d_img = {
        k: v.clone()
        for k, v in model.embed_image(
            rearrange(torch.from_numpy(rgb), "h w c -> 1 c h w")
            .float()
            .div(127.5)
            .sub(1.0)
            .to(device)
        ).items()
    }

    all_flows = []
    probe_points_pixel = np.zeros((N, num_offsets, 2, 2))  # (objects, dirs, start/end, xy)

    for i in range(N):
        centroid = centroids_norm[i:i + 1]  # [1, 2]

        for j in range(num_offsets):
            # poke pos + flow
            poke_start = centroid
            poke_end = centroid + torch.stack([dx[j], dy[j]], dim=0)[None]

            poke_pos = poke_start.unsqueeze(0)  # [1, 1, 2]
            poke_flow = (poke_end - poke_start).unsqueeze(0)  # [1, 1, 2]

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                # batch
                num_queries = query_pos.shape[1]
                flow_chunks = []

                for q_idx in range(0, num_queries, query_batch_size):
                    query_batch = query_pos[:, q_idx:q_idx + query_batch_size]

                    pred = model.predict_parallel(
                        poke_pos=poke_pos,
                        poke_flow=poke_flow,
                        query_pos=query_batch,
                        camera_static=True,
                        d_img=d_img
                    )
                    flow_chunks.append(pred.mean)

                flow_pred = torch.cat(flow_chunks, dim=1)
                flow = rearrange(
                    flow_pred,
                    "b (h w) c -> b h w c",
                    h=flow_resolution,
                    w=flow_resolution
                )

            all_flows.append(flow)

            # pixel coords
            drag_start_pixel = normalized_to_pixel(poke_start, (target_size, target_size))
            drag_end_pixel = normalized_to_pixel(poke_end, (target_size, target_size))

            probe_points_pixel[i, j, 0] = [drag_start_pixel[0, 0].item(), drag_start_pixel[0, 1].item()]
            probe_points_pixel[i, j, 1] = [drag_end_pixel[0, 0].item(), drag_end_pixel[0, 1].item()]

    flows = torch.cat(all_flows, dim=0)  # [N*num_offsets, H, W, 2]

    return flows, probe_points_pixel


# for parallel
def run_flow_generation(model, h5_files, output_dir, min_prob, num_offsets,
                        flow_resolution, min_mag, max_mag, device, overwrite,
                        gpu_id=None, max_vis=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = output_dir / 'vis'
    vis_dir.mkdir(exist_ok=True)

    gpu_prefix = f"[GPU {gpu_id}] " if gpu_id is not None else ""
    vis_counter = 0

    for h5_path in tqdm(h5_files, desc=f"{gpu_prefix}Processing"):
        try:
            base_name = h5_path.stem
            output_path = output_dir / f"{base_name}.h5"

            if output_path.exists() and not overwrite:
                print(f"{gpu_prefix}Skipping {base_name} (already exists)")
                continue

            # Load input data
            with h5py.File(h5_path, 'r') as f_in:
                image = f_in['image'][:]
                prob_motion = f_in['prob_motion'][:]
                # Get sampled points for specified min_prob
                min_prob_key = f'min_prob_{min_prob}'
                sampled_points = f_in['sampled_points'][min_prob_key][:]

            if len(sampled_points) == 0:
                print(f"{gpu_prefix}No sampled points for {base_name}, skipping...")
                continue

            target_size = 448 # input for model
            H, W = image.shape[:2]
            image_resized = np.array(Image.fromarray(image).resize((target_size, target_size), Image.BICUBIC))

            # sampled points from 64x64 grid -> image cords
            prob_H, prob_W = prob_motion.shape
            sampled_points_img = sampled_points.copy().astype(float)
            sampled_points_img[:, 0] = sampled_points[:, 0] * (W / prob_W)
            sampled_points_img[:, 1] = sampled_points[:, 1] * (H / prob_H)

            # scale to resized target img
            sampled_points_resized = sampled_points_img.copy()
            sampled_points_resized[:, 0] = sampled_points_img[:, 0] * (target_size / W)
            sampled_points_resized[:, 1] = sampled_points_img[:, 1] * (target_size / H)

            # pts in [0, 1]
            centroids_norm = torch.from_numpy(sampled_points_resized / target_size).float().to(device)

            # gen flows
            flows, probe_points_pixel = predict_flows_poke_pos(
                model=model,
                rgb=image_resized,
                centroids_norm=centroids_norm,
                num_offsets=num_offsets,
                flow_resolution=flow_resolution,
                min_mag=min_mag / 256.0,
                max_mag=max_mag / 256.0,
                device=torch.device(device),
                query_batch_size=4096,
                target_size=target_size
            )
            flows_np = flows.cpu().numpy()

            N = len(sampled_points) # (N, num_offsets, H, W, 2)
            flows_np = flows_np.reshape(N, num_offsets, flow_resolution, flow_resolution, 2)

            # save
            with h5py.File(output_path, 'w') as f_out:
                f_out.create_dataset('image', data=image, compression='gzip')
                f_out.create_dataset('prob_motion', data=prob_motion, compression='gzip')
                f_out.create_dataset('sampled_points', data=sampled_points, compression='gzip')
                f_out.create_dataset('flows', data=flows_np, compression='gzip')
                f_out.create_dataset('probe_points', data=probe_points_pixel, compression='gzip')

                f_out.attrs['num_offsets'] = num_offsets
                f_out.attrs['flow_resolution'] = flow_resolution
                f_out.attrs['min_mag'] = min_mag
                f_out.attrs['max_mag'] = max_mag
                f_out.attrs['min_prob_used'] = min_prob
                f_out.attrs['target_size'] = target_size

            if vis_counter < max_vis: # vis
                vis_path = vis_dir / f"{base_name}_flows.jpg"
                visualize_flows_grid(flows_np, sampled_points, image, save_path=vis_path)
                vis_counter += 1

        except Exception as e:
            print(f"{gpu_prefix} has error processing {h5_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue


def main(args): # for single gpu
    print("Loading FPT model")
    model = torch.hub.load(".", "fpt_base", source="local")
    model.eval()
    model.to(args.device)

    # Get all H5 files
    input_dir = Path(args.input_dir)
    h5_files = sorted(input_dir.glob('*.h5'))
    print(f"Found {len(h5_files)} H5 files to process")

    run_flow_generation(
        model=model,
        h5_files=h5_files,
        output_dir=args.output_dir,
        min_prob=args.min_prob,
        num_offsets=args.num_offsets,
        flow_resolution=args.flow_resolution,
        min_mag=args.min_mag,
        max_mag=args.max_mag,
        device=args.device,
        overwrite=args.overwrite,
        gpu_id=None,
        max_vis=10
    )

    print(f"\n{'=' * 80}")
    print("Processing complete!")
    print(f"Output saved to: {args.output_dir}")
    print(f"{'=' * 80}")

def visualize_flows_grid(flows, sampled_points, image, save_path=None):
    from torchvision.utils import flow_to_image

    num_points = flows.shape[0]
    n_cols = min(num_points, 5)
    n_rows = (num_points + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axs = np.expand_dims(axs, 0)
    if n_cols == 1:
        axs = np.expand_dims(axs, 1)

    # Scale sampled points to image size
    H, W = image.shape[:2]
    sampled_points_scaled = sampled_points.copy().astype(float)
    sampled_points_scaled[:, 0] = sampled_points[:, 0] * (W / 64.0)  # x
    sampled_points_scaled[:, 1] = sampled_points[:, 1] * (H / 64.0)  # y

    for idx in range(num_points):
        # first offset flow
        flow = flows[idx, 0]  # (H, W, 2)

        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()  # (2, H, W) - convert to float32
        flow_rgb = flow_to_image(flow_tensor).permute(1, 2, 0).numpy()  # (H, W, 3)

        x_probe, y_probe = sampled_points_scaled[idx].astype(int)

        row, col = divmod(idx, n_cols)
        ax = axs[row][col]

        ax.imshow(flow_rgb)

        ax.scatter(x_probe, y_probe, color='lime', s=100, marker='x', linewidths=3)
        ax.set_title(f'Point {idx} ({sampled_points[idx, 0]}, {sampled_points[idx, 1]})',
                     fontweight='bold')
        ax.axis('off')

    for j in range(num_points, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axs[row][col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate flows from P motion sampled points')
    parser.add_argument('--input_dir', type=str, default='/ccn2/u/lilianch/external_repos/flow-poke-transformer/autosegp/pmotion12_minprob_samples/h5',
                        help='Directory containing input H5 files with prob_motion and sampled_points')
    parser.add_argument('--output_dir', type=str, default='/ccn2/u/lilianch/external_repos/flow-poke-transformer/autosegp/flows_pmotion12_0.005minprob_8dist/')
    parser.add_argument('--min_prob', type=float, default=0.005,
                        help='Which min_prob threshold to use for sampled points')
    parser.add_argument('--num_offsets', type=int, default=15)
    parser.add_argument('--flow_resolution', type=int, default=256)
    parser.add_argument('--min_mag', type=float, default=10.0)
    parser.add_argument('--max_mag', type=float, default=25.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()
    main(args)