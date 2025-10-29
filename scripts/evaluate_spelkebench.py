import os
import math
import argparse
from pathlib import Path
from typing import Literal, Tuple, List
from PIL import Image
import h5py as h5
import numpy as np
import torch
import cv2
from einops import rearrange
from torch.distributions import MixtureSameFamily
import matplotlib.pyplot as plt
from matplotlib import patches
from tqdm import tqdm
from flow_poke.model import FlowPokeTransformer, FlowPokeTransformer_Base, make_axial_pos_2d, MixtureSameFamily
from scripts.utils import set_seed, offset_multiple_centroids, pixel_to_normalized, normalized_to_pixel, get_dot_product_map, threshold_heatmap


def predict_flows(
        model,
        rgb: torch.Tensor,
        centroids_norm: torch.Tensor, #Normalized centroids [N, 2] in (x, y) format
        num_offsets: int,  #Number of directions per centroid
        flow_resolution: int,
        prediction_mode: Literal["parallel", "autoregressive"], #parallel or autoregressive
        ar_downsampling_factor: int = 4,#AUTOREGRESSIVE ONLY
        ar_num_generations: int = 3,#AUTOREGRESSIVE ONLY
        min_mag: float = 10.0/256.0, # into 0,1 space
        max_mag: float = 25.0/256.0, # into 0,1 space
        device: torch.device = None,
        query_batch_size: int = 4096,
        target_size=448
) -> Tuple[torch.Tensor, List[Tuple[float, float, float, float]]]:
    N = centroids_norm.shape[0]

    # Create query positions
    query_pos = make_axial_pos_2d(flow_resolution, flow_resolution, device=device)[None]  # [1, H*W, 2]

    dx, dy = offset_multiple_centroids(centroids_norm, num_offsets, min_mag, max_mag)

    img_shape = (rgb.shape[0], rgb.shape[1])

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

    # normalize magnitudes relative to flow res TODO: see if needed
    # dx_norm = dx / (flow_resolution / 2)  # Scale to normalized space
    # dy_norm = dy / (flow_resolution / 2)
    # print(f"normalized to flow res dx, dy: {dx_norm}, {dy_norm}")

    all_flows = []
    all_drags_pixel = []

    if prediction_mode == "parallel":
        for i in range(N): # each segment
            centroid = centroids_norm[i:i + 1]  # [1, 2]

            for j in range(num_offsets):
                # Create poke position and flow
                poke_start = centroid  # [1, 2]
                poke_end = centroid + torch.stack([dx[j], dy[j]], dim=0)[None]  # [1, 2]

                poke_pos = poke_start  # [1, 2]
                poke_flow = poke_end - poke_start  # [1, 2]

                poke_pos = poke_pos.unsqueeze(0)  # [1, 1, 2]
                poke_flow = poke_flow.unsqueeze(0)  # [1, 1, 2]

                # from dempo:
                # poke_pos: tensor([[[0.7388, 0.1116]]], device='cuda:0')
                # poke_flow: tensor([[[0.1004, 0.0223]]], device='cuda:0') for top right of img
                # print(f'poke_pos: {poke_pos}')
                # print(f'poke_flow: {poke_flow}')

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    # batch queries
                    num_queries = query_pos.shape[1]
                    flow_chunks = []

                    for i in range(0, num_queries, query_batch_size):
                        query_batch = query_pos[:, i:i + query_batch_size]

                        pred: MixtureSameFamily = model.predict_parallel(
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

                # drag -> pixel coords
                drag_start_pixel = normalized_to_pixel(poke_start,
                                                       (target_size, target_size))  # Use target_size, not original
                drag_end_pixel = normalized_to_pixel(poke_end, (target_size, target_size))
                drag_pixel = (
                    drag_start_pixel[0, 0],
                    drag_start_pixel[0, 1],
                    drag_end_pixel[0, 0],
                    drag_end_pixel[0, 1]
                )

                all_drags_pixel.append(drag_pixel)

        flows = torch.cat(all_flows, dim=0)  # [N*num_offsets, H, W, 2]

    elif prediction_mode == "autoregressive":
        ar_flow_resolution = flow_resolution // ar_downsampling_factor
        query_pos_ar = make_axial_pos_2d(ar_flow_resolution, ar_flow_resolution, device=device)[None]

        for i in range(N): # n segments
            centroid = centroids_norm[i:i + 1]  # [1, 2]

            for j in range(num_offsets): # num offsets
                # Create poke position and flow
                poke_start = centroid
                poke_end = centroid + torch.stack([dx_norm[j], dy_norm[j]], dim=0)[None]

                poke_pos = poke_start.unsqueeze(0)  # [1, 1, 2]
                poke_flow = (poke_end - poke_start).unsqueeze(0)  # [1, 1, 2]

                for gen in range(ar_num_generations): # num generations (randomized)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        flow_ar = rearrange(
                            model.predict_autoregressive(
                                poke_pos=poke_pos,
                                poke_flow=poke_flow,
                                query_pos=query_pos_ar,
                                camera_static=True,
                                d_img=d_img,
                                randomize_order=True,
                            ),
                            "b (h w) c -> b h w c",
                            h=ar_flow_resolution,
                            w=ar_flow_resolution,
                        )  # [1, H_ar, W_ar, 2]

                        # Upsample using parallel prediction
                        if ar_downsampling_factor != 1:
                            num_queries = query_pos.shape[1] # batch
                            flow_chunks = []

                            for i in range(0, num_queries, query_batch_size):
                                query_batch = query_pos[:, i:i + query_batch_size]

                                pred = model.predict_parallel(
                                    poke_pos=torch.cat([poke_pos, query_pos_ar], dim=1),
                                    poke_flow=torch.cat(
                                        [poke_flow, rearrange(flow_ar, "b h w c -> b (h w) c")],
                                        dim=1
                                    ),
                                    query_pos=query_batch,
                                    camera_static=True,
                                    d_img=d_img,
                                )
                                flow_chunks.append(pred.mean)

                            flow = rearrange(
                                torch.cat(flow_chunks, dim=1),
                                "b (h w) c -> b h w c",
                                h=flow_resolution,
                                w=flow_resolution,
                            )
                        else:
                            flow = flow_ar

                    all_flows.append(flow)

                # drag same for all gens, so store only once
                drag_start_pixel = normalized_to_pixel(poke_start, img_shape)
                drag_end_pixel = normalized_to_pixel(poke_end, img_shape)
                drag_pixel = (
                    drag_start_pixel[0, 0],
                    drag_start_pixel[0, 1],
                    drag_end_pixel[0, 0],
                    drag_end_pixel[0, 1]
                )
                for _ in range(ar_num_generations):
                    all_drags_pixel.append(drag_pixel)

        flows = torch.cat(all_flows, dim=0)  # [N*num_offsets*ar_num_generations, H, W, 2]

    return flows, all_drags_pixel


def segment_from_flows(
        flows: torch.Tensor,
        drags_pixel: List[Tuple[float, float, float, float]],
        num_flows_per_segment: int,
) -> np.ndarray:
    """
    Create segmentation mask from flows using dot product w/ drag dirs
    """
    all_dot_prods = []

    for i in range(num_flows_per_segment):
        flow = flows[i]  # [H, W, 2]
        drag = drags_pixel[i]
        dot_product_map = get_dot_product_map(flow, drag)
        all_dot_prods.append(dot_product_map)

    # Average dot products across all flows
    all_dot_prods = torch.stack(all_dot_prods, dim=0)
    mean_dot_prod = all_dot_prods.mean(dim=0)
    mean_dot_prod_np = mean_dot_prod.cpu().numpy()

    # Threshold to get binary mask
    segment = threshold_heatmap(mean_dot_prod_np)

    return segment


def visualize_results(
        rgb: np.ndarray,
        centroid_pixel: np.ndarray,
        drags_pixel: List[Tuple[float, float, float, float]],
        flows: torch.Tensor,
        dot_prod_example: np.ndarray,
        pred_segment: np.ndarray,
        gt_segment: np.ndarray,
        save_path: Path,
        num_flows_to_show: int = 2,
):
    """
    Visualize evaluation results with flow directions.
    """
    from torchvision.utils import flow_to_image

    # drags are in 448x448 space (resized image)
    # rgb is (256x256) (+centrpod)
    # flows/dotprods at flow res (256x256)

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    rgb_h, rgb_w = rgb.shape[:2]
    target_size = 448  # Size drags are in
    flow_h, flow_w = flows.shape[1:3]  # Flow resolution

    scale_x = rgb_w / target_size
    scale_y = rgb_h / target_size

    centroid_rgb = centroid_pixel.copy()

    drags_rgb = []
    for drag in drags_pixel:
        x1, y1, x2, y2 = drag
        drags_rgb.append((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y))

    scale_x_flow = flow_w / target_size
    scale_y_flow = flow_h / target_size

    drags_flow = []
    for drag in drags_pixel:
        x1, y1, x2, y2 = drag
        drags_flow.append((x1 * scale_x_flow, y1 * scale_y_flow,
                           x2 * scale_x_flow, y2 * scale_y_flow))

    # Row 1, Col 1: RGB with drags
    axes[0, 0].imshow(rgb)
    axes[0, 0].plot(centroid_rgb[0], centroid_rgb[1], 'r*', markersize=15, label='Centroid')
    for i, drag in enumerate(drags_rgb[:num_flows_to_show]):
        x1, y1, x2, y2 = drag
        axes[0, 0].arrow(x1, y1, x2 - x1, y2 - y1, color='yellow', width=2,
                         head_width=10, head_length=10, alpha=0.7)
    axes[0, 0].set_title('RGB with Drags')
    axes[0, 0].axis('off')
    axes[0, 0].legend()

    # Row 1, Col 2: Predicted segment
    axes[0, 1].imshow(rgb)
    axes[0, 1].imshow(pred_segment, alpha=0.5, cmap='Reds')
    axes[0, 1].set_title('Predicted Segment')
    axes[0, 1].axis('off')

    # Row 1, Col 3: Ground truth segment
    axes[0, 2].imshow(rgb)
    axes[0, 2].imshow(gt_segment, alpha=0.5, cmap='Greens')
    axes[0, 2].set_title('Ground Truth Segment')
    axes[0, 2].axis('off')

    # Row 1, Col 4: Overlay comparison
    axes[0, 3].imshow(rgb)
    overlap = np.zeros((*pred_segment.shape, 3))
    overlap[pred_segment == 1, 0] = 1  # Red: predicted
    overlap[gt_segment == 1, 1] = 1  # Green: ground truth
    axes[0, 3].imshow(overlap, alpha=0.5)
    axes[0, 3].set_title('Overlay (Yellow=Both, Red=FP, Green=FN)')
    axes[0, 3].axis('off')

    # Row 2: Flow 1
    # Col 1: Flow 1 magnitude
    flow1 = flows[0].cpu().float()  # [H, W, 2]
    flow1_mag = torch.norm(flow1, dim=-1).numpy()
    im = axes[1, 0].imshow(flow1_mag, cmap='jet')
    plt.colorbar(im, ax=axes[1, 0])
    axes[1, 0].set_title('Flow 1 Magnitude')
    axes[1, 0].axis('off')

    # Col 2: Flow 1 direction (RGB visualization)
    flow1_rgb = flow_to_image(flow1.permute(2, 0, 1)).permute(1, 2, 0).numpy()
    axes[1, 1].imshow(flow1_rgb)
    axes[1, 1].set_title('Flow 1 Direction')
    axes[1, 1].axis('off')

    # Col 3: Dot product map (flow 1) + direction
    dot_prod1 = get_dot_product_map(flows[0], drags_pixel[0]).cpu().numpy()
    im = axes[1, 2].imshow(dot_prod1, cmap='RdBu_r')
    # Overlay drag arrow in flow space
    x1, y1, x2, y2 = drags_flow[0]
    axes[1, 2].arrow(x1, y1, x2 - x1, y2 - y1, color='yellow',
                     width=2, head_width=8, alpha=0.8)
    plt.colorbar(im, ax=axes[1, 2])
    axes[1, 2].set_title('Dot Product (Flow 1)')
    axes[1, 2].axis('off')

    # Col 4: Empty
    axes[1, 3].axis('off')

    # Row 3: Flow 2
    if flows.shape[0] > 1:
        # Col 1: Flow 2 magnitude
        flow2 = flows[1].cpu().float()  # [H, W, 2]
        flow2_mag = torch.norm(flow2, dim=-1).numpy()
        im = axes[2, 0].imshow(flow2_mag, cmap='jet')
        plt.colorbar(im, ax=axes[2, 0])
        axes[2, 0].set_title('Flow 2 Magnitude')
        axes[2, 0].axis('off')

        # Col 2: Flow 2 direction (RGB visualization)
        flow2_rgb = flow_to_image(flow2.permute(2, 0, 1)).permute(1, 2, 0).numpy()
        axes[2, 1].imshow(flow2_rgb)
        axes[2, 1].set_title('Flow 2 Direction')
        axes[2, 1].axis('off')

        # Col 3: Dot product map (flow 2) + direction
        dot_prod2 = get_dot_product_map(flows[1], drags_pixel[1]).cpu().numpy()
        im = axes[2, 2].imshow(dot_prod2, cmap='RdBu_r')
        # Overlay drag arrow in flow space
        x1, y1, x2, y2 = drags_flow[1]
        axes[2, 2].arrow(x1, y1, x2 - x1, y2 - y1, color='yellow',
                         width=2, head_width=8, alpha=0.8)
        plt.colorbar(im, ax=axes[2, 2])
        axes[2, 2].set_title('Dot Product (Flow 2)')
        axes[2, 2].axis('off')
    else:
        axes[2, 0].axis('off')
        axes[2, 1].axis('off')
        axes[2, 2].axis('off')

    # Col 4: Empty
    axes[2, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_spelkebench(
        dataset_path: str,
        output_dir: str,
        prediction_mode: Literal["parallel", "autoregressive"] = "parallel",
        num_offsets: int = 4,
        flow_resolution: int = 256,
        ar_downsampling_factor: int = 4, # AUTOREGRESSIVE OLY
        ar_num_generations: int = 3, # AUTOREGRESSIVE OLY
        seed: int = 42,
        device: str = "cuda",
        max_examples: int = None,
        start_idx: int = None,
        end_idx: int = None,
        model=None
):
    set_seed(seed)
    device = torch.device(device)

    print("Loading FlowPokeTransformer model...")
    if not model:
        model = torch.hub.load(".", "fpt_base", source="local")
    model.eval()
    model.to(device)
    model.requires_grad_(False)

    # output directories
    output_dir = Path(output_dir)
    results_dir = output_dir / "results"
    vis_dir = output_dir / "vis"
    results_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Open dataset
    with h5.File(dataset_path, 'r') as data:
        # Get all example keys (filter for entityseg or openx)
        all_keys = [k for k in data.keys() if k.startswith('entityseg_') or k.startswith('openx_')]

        if max_examples:
            all_keys = all_keys[:max_examples]
        if start_idx is not None and end_idx is not None:
            all_keys = all_keys[start_idx:end_idx]
            print(f"Processing subset: {start_idx} to {end_idx}")

        print(f"Processing {len(all_keys)} examples...")

        for example_idx, example_key in enumerate(tqdm(all_keys)):
            example_data = data[example_key]

            # Load data
            rgb = example_data['rgb'][:]  # [H, W, 3]
            segments = example_data['segment'][:]  # [N, H, W]
            centroids = example_data['centroid'][:]  # [N, 2] in (x, y) format
            filename = example_data['filename'][()].decode() if 'filename' in example_data else example_key

            N_segments = segments.shape[0]
            N_centroids = centroids.shape[0]

            # Handle mismatch between centroids and segments
            if N_centroids != N_segments:
                print(
                    f"WARNING: {N_centroids} centroids but {N_segments} segments. Using min: {min(N_centroids, N_segments)}")
                N = min(N_centroids, N_segments)
                centroids = centroids[:N]
                segments = segments[:N]
            else:
                N = N_segments

            H, W = rgb.shape[:2]
            print(f"\n=== Processing {example_key} ===")
            print(f"RGB shape: {rgb.shape}, Centroids: {centroids}")
            # Resize image
            target_size = 448
            H, W = rgb.shape[:2]
            image_resized = np.array(Image.fromarray(rgb).resize((target_size, target_size), Image.BICUBIC))

            # original coords -> resized coords -> normalized [0, 1]
            centroids_resized = centroids.copy()
            centroids_resized[:, 0] = centroids[:, 0] * (target_size / W)  # x
            centroids_resized[:, 1] = centroids[:, 1] * (target_size / H)  # y

            # Normalize to [0, 1]
            centroids_norm_np = np.zeros((N, 2), dtype=np.float32)
            centroids_norm_np[:, 0] = centroids_resized[:, 0] / target_size  # x in [0, 1]
            centroids_norm_np[:, 1] = centroids_resized[:, 1] / target_size  # y in [0, 1]
            centroids_norm = torch.from_numpy(centroids_norm_np).to(device)

            # Storage for all results
            all_flows_per_segment = []
            all_drags_per_segment = []
            all_pred_segments = []

            for seg_idx in range(N): # per seg
                centroid_norm = centroids_norm[seg_idx:seg_idx + 1]  # [1, 2]
                gt_segment = segments[seg_idx]  # [H, W]

                # get flows
                flows, drags_pixel = predict_flows(
                    model=model,
                    rgb=image_resized,
                    centroids_norm=centroid_norm,
                    num_offsets=num_offsets,
                    flow_resolution=flow_resolution,
                    prediction_mode=prediction_mode,
                    ar_downsampling_factor=ar_downsampling_factor,
                    ar_num_generations=ar_num_generations,
                    device=device,
                    query_batch_size=4096,
                )

                if prediction_mode == "parallel":
                    num_flows = num_offsets
                else:  # autoregressive
                    num_flows = num_offsets * ar_num_generations

                # Generate segmentation
                pred_segment = segment_from_flows(flows, drags_pixel, num_flows)

                if pred_segment.shape != gt_segment.shape: # resize for comparison (256,256)
                    pred_segment = cv2.resize(
                        pred_segment.astype(np.uint8),
                        (W, H),
                        interpolation=cv2.INTER_NEAREST
                    )

                all_flows_per_segment.append(flows.cpu().numpy())
                all_drags_per_segment.append(drags_pixel)
                all_pred_segments.append(pred_segment)

                dot_prod_example = get_dot_product_map(flows[0], drags_pixel[0]).cpu().numpy()

                vis_path = vis_dir / f"{example_key}_seg{seg_idx}.png"
                visualize_results(
                    rgb=rgb,
                    centroid_pixel=centroids[seg_idx],
                    drags_pixel=drags_pixel[:num_offsets],  # unique drags
                    flows=flows[:num_offsets],  # first few flows
                    dot_prod_example=dot_prod_example,
                    pred_segment=pred_segment,
                    gt_segment=gt_segment,
                    save_path=vis_path,
                )

                # After predict_flows call
                print(f"  Generated {len(drags_pixel)} drags")

            # save
            result_path = results_dir / f"{example_key}.h5"
            with h5.File(result_path, 'w') as f:
                # og data
                f.create_dataset('centroid', data=centroids)
                f.create_dataset('filename', data=filename)
                f.create_dataset('rgb', data=rgb)
                f.create_dataset('segment_gt', data=segments)

                # predictions
                for seg_idx in range(N):
                    seg_group = f.create_group(f'segment_{seg_idx}')

                    flows_array = all_flows_per_segment[seg_idx]  # [num_flows, H, W, 2]
                    seg_group.create_dataset('flows', data=flows_array)

                    # drags as array
                    drags_array = np.array(all_drags_per_segment[seg_idx])
                    seg_group.create_dataset('drags', data=drags_array)

                    # pred segment
                    seg_group.create_dataset('segment_pred', data=all_pred_segments[seg_idx])

            if (example_idx + 1) % 10 == 0:
                print(f"Processed {example_idx + 1}/{len(all_keys)} examples")

    print(f"Doneee ! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpelkeBench dataset")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prediction_mode", type=str, default="parallel",
                        choices=["parallel", "autoregressive"])
    parser.add_argument("--num_offsets", type=int, default=24)
    parser.add_argument("--flow_resolution", type=int, default=256)
    parser.add_argument("--ar_downsampling_factor", type=int, default=4, help="only for autoregressive mode")
    parser.add_argument("--ar_num_generations", type=int, default=3, help="only for autoregressive mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_examples", type=int, default=None, help="max examples to process (for slow AR rollouts)")

    parser.add_argument("--start_idx", type=int, default=None, help="max examples to process (for slow AR rollouts)")
    parser.add_argument("--end_idx", type=int, default=None, help="max examples to process (for slow AR rollouts)")


    args = parser.parse_args()

    # Run evaluation
    evaluate_spelkebench(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        prediction_mode=args.prediction_mode,
        num_offsets=args.num_offsets,
        flow_resolution=args.flow_resolution,
        ar_downsampling_factor=args.ar_downsampling_factor,
        ar_num_generations=args.ar_num_generations,
        seed=args.seed,
        device=args.device,
        max_examples=args.max_examples,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )


if __name__ == "__main__":
    main()