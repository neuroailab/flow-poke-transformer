import cv2
import numpy as np

def evaluate_AP_AR_single_image(pred_segments, gt_segments): # add things to https://github.com/neuroailab/ccwm/blob/a7cbc877e6c2d055c6eb1f489a72a3c4f3ee2535/ccwm/utils/iterative_affinity_segment.py#L162
    iou_mat = batched_iou(gt_segments, pred_segments)
    gt_inds, pred_inds = linear_sum_assignment(1. - iou_mat)

    ious = iou_mat[gt_inds, pred_inds]

    num_gt_segments = gt_segments.shape[0]
    num_pred_segments = pred_segments.shape[0]

    precisions = []
    recalls = []

    thresholds = np.arange(start=0.50, stop=0.95, step=0.05)

    for i, iou_thresh in enumerate(thresholds):
        tp = np.count_nonzero(ious >= iou_thresh)

        if num_pred_segments == 0:
            precisions.append(0)
        else:
            precisions.append(tp / num_pred_segments)

        if num_gt_segments == 0:
            recalls.append(0)
        else:
            recalls.append(tp / num_gt_segments)

    return {
        'AP': np.mean(precisions),
        'AR': np.mean(recalls),
        'assignments': [gt_inds, pred_inds],
        'iou_mat': iou_mat,
        'thresholds': thresholds
    }

def evaluate_AP_AR_single_image(pred_segments, gt_segments):
    """
    Compute Average Precision (AP) and Average Recall (AR) for a single image.
    Precision and Recall are computed over IoU=.50:.05:.95.
    """
    iou_mat = batched_iou(gt_segments, pred_segments)
    gt_inds, pred_inds = linear_sum_assignment(1. - iou_mat)
    ious = np.array(iou_mat[gt_inds, pred_inds])
    num_gt_segments = gt_segments.shape[0]
    num_pred_segments = pred_segments.shape[0]
    precisions = []
    f1_scores = []
    recalls = []

    thresholds = np.arange(start=0.50, stop=0.95, step=0.05)
    for i, iou_thresh in enumerate(thresholds):
        tp = np.count_nonzero(ious >= iou_thresh)
        if num_pred_segments == 0:
            precisions.append(0)
        else:
            precisions.append(tp / num_pred_segments)
        if num_gt_segments == 0:
            recalls.append(0)
        else:
            recalls.append(tp / num_gt_segments)

        # add for f1
        p = precisions[-1]
        r = recalls[-1]
        if p + r > 0:
            f1_scores.append(2 * p * r / (p + r))
        else:
            f1_scores.append(0)
    return {
        'AP': np.mean(precisions),
        'AR': np.mean(recalls),
        'F1': np.mean(f1_scores),
        'assignments': [gt_inds, pred_inds],
        'iou_mat': iou_mat,
        'mean_iou': ious.mean() if len(ious) > 0 else 0.0,  # Mean IoU of matched pairs
        'thresholds': thresholds,
    }


import os
from scripts.utils import process_image_for_model
from flow_poke.model import make_axial_pos_2d
import torch
import h5py as h5
from PIL import Image
import cv2
from einops import rearrange
from scipy.ndimage import minimum_filter
from scipy.spatial.distance import cdist
import numpy as np
IMAGE_SIZE=256
from tqdm import tqdm

def sample_diverse_high_prob_points(prob_map, num_samples=16, min_prob=0.001, min_dist=8):
    H, W = prob_map.shape
    flat_probs = prob_map.flatten()
    valid = np.where(flat_probs >= min_prob)[0]
    sorted_idxs = valid[np.argsort(-flat_probs[valid])]
    sampled = []
    for idx in sorted_idxs:
        if len(sampled) >= num_samples:
            break
        y, x = divmod(idx, W)
        pt = np.array([x, y])
        if not sampled or np.min(cdist([pt], sampled)) >= min_dist:
            sampled.append(pt)
    return np.array(sampled)

def compute_motion_maps(image, model, grid_size, threshold=4, device='cuda'):
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
    from scripts.utils import render_density_grid
    pred_single = pred[0]  # Remove batch dimension, torch.Size([4096, 4])

    chunk_size = 1024  # Process 1024 query points at a time
    num_points = grid_size * grid_size
    dense_prob_maps = torch.zeros(num_points, grid_size, grid_size, device=device, dtype=torch.float16)  # Use float16!

    for chunk_start in tqdm(range(0, num_points, chunk_size), desc="Computing dense maps"):
        chunk_end = min(chunk_start + chunk_size, num_points)

        for i in range(chunk_start, chunk_end):
            query_pos = coords[i]
            x_min, x_max = 0 - query_pos[0], 1 - query_pos[0]
            y_min, y_max = 0 - query_pos[1], 1 - query_pos[1]

            out = 0
            for j in range(4):
                prob = pred_single[i].mixture_distribution.probs[j].item()
                out = out + (prob * render_density_grid(
                    pred_single[i].component_distribution[j], x_min, x_max, y_min, y_max,
                    grid_resolution=grid_size
                ))
            out = out / (out.sum() + 1e-8)
            dense_prob_maps[i] = out.half()  # Store as float16

    # Compute all flows at once in [0,1] space, then scale to pixels
    # coords_grid: (grid_size, grid_size, 2), coords: (grid_size^2, 2)
    flows_to_all = (coords_grid[:, :, None, :] - coords[None, None, :, :]) * 256 # (grid, grid, grid^2, 2) in [0,1] space
    flows_to_all_pixels = flows_to_all # * IMAGE_SIZE  # NOW in pixel space
    flow_mags = torch.norm(flows_to_all_pixels, dim=-1)  # (grid, grid, grid^2) in PIXELS

    # Permute dense_prob_maps from (grid^2, grid, grid) to (grid, grid, grid^2) for broadcasting
    dense_prob_maps_permuted = dense_prob_maps.permute(1, 2, 0)  # (grid, grid, grid^2)

    motion_mask = (flow_mags > threshold).float()  # threshold is in PIXELS now

    # Sum over spatial dimensions (grid, grid) to get motion prob for each query point
    prob_motion_map = (dense_prob_maps_permuted * motion_mask).sum(dim=(0, 1))  # (grid^2,)
    prob_motion_heatmap = prob_motion_map.reshape(grid_size, grid_size).cpu().numpy()

    # Expected magnitude
    expected_magnitude_map = (dense_prob_maps_permuted * flow_mags).sum(dim=(0, 1))  # (grid^2,)
    expected_magnitude_map = expected_magnitude_map.reshape(grid_size, grid_size).cpu().numpy()
    return expected_magnitude_map, prob_motion_heatmap

def sample_pmotion(im, num_samples, min_prob, min_dist, pm, min_filter=False):
    H, W, _ = im.shape
    samples_all = []
    prob_maps = []
    crops = []
    samples_all_crop = []

    # print("Loading FPT model")
    # model = torch.hub.load(".", "fpt_base", source="local")
    # model.eval()
    # model.to('cuda')

    n_tries = 4

    # for i, (dy, dx) in enumerate(offsets):
    crop = im  # [dy:dy + h_half, dx:dx + w_half]

    # resize crop to 256 x 256 using cv2
    crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
    crop = crop.astype(np.uint8)
    crop = Image.fromarray(crop.astype(np.uint8))
    crops.append(crop)

    # pm = compute_motion_maps(crop, model, 64, 25, 'cuda')

    kernel_size = 3  # or use an odd size like 5 if you want exact centering
    prob_map_filt = minimum_filter(pm, size=kernel_size, mode='reflect')

    for i in range(n_tries):

        if (min_filter) and (i == 0):
            prob_map = prob_map_filt
        else:
            prob_map = pm

        prob_maps.append(prob_map)

        grid_pts = sample_diverse_high_prob_points(prob_map, num_samples=num_samples, min_dist=min_dist, min_prob=min_prob)

        if len(grid_pts) != 0:
            img_pts = (grid_pts * (H // 64)).astype(int)
            samples_all.append(img_pts)
            samples_all_crop.append(grid_pts * 4)
        else:
            samples_all_crop.append(np.array([]))
        if len(samples_all) !=0 and (len(np.vstack(samples_all)) >= num_samples):
            break
        else:
            min_dist = min_dist / 2

    if len(samples_all) == 0:
        pts = np.array(samples_all)
    else:
        pts = np.vstack(samples_all)

        #make unique
        pts = np.unique(pts, axis=0)
        # if len(pts) > num_samples:
        #     pts = pts[:num_samples]

    return pts, prob_maps, crops, samples_all_crop
