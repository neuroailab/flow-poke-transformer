import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cdist
import json
import os
import math
import torch
from PIL import ImageOps, Image
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
import numpy as np
from torch.nn import functional
from pycocotools import mask as mask_util
import torch.nn.functional as F

def batched_iou(x, y=None):
    """IoU between (B, H, W)"""
    if y is None:
        y = x

    xp = x[:, None]
    yp = y[None]

    intersection = (xp & yp).sum(axis=(-1, -2))
    union = (xp | yp).sum(axis=(-1, -2))

    return intersection / union

def map_resized_coord_to_original(x_256, y_256, x1, y1, x2, y2):
    w = min(x2, 256) - x1
    h = min(y2, 256) - y1
    x_orig = x1 + (x_256 / 256.0) * w
    y_orig = y1 + (y_256 / 256.0) * h
    return int(x_orig), int(y_orig)


def plot_flows_with_probes(flow_imgs, probe_points, save_path=None):
    n = len(flow_imgs)
    n_cols = min(n, 5)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if n == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = np.expand_dims(axs, 0)
    elif n_cols == 1:
        axs = np.expand_dims(axs, 1)

    for i, (flow_img, pt) in enumerate(zip(flow_imgs, probe_points)):
        row, col = divmod(i, n_cols)
        ax = axs[row][col]
        ax.imshow(flow_img)
        ax.scatter(*pt, color='black', s=40)
        ax.set_title(f'Object {i}')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.axis('off')

    for j in range(i + 1, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axs[row][col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def compute_affinity_matrix(flat_data, device, batch_size):
    N = flat_data.shape[0]
    affinity_matrix = torch.zeros((N, N), dtype=torch.float32, device=device)
    for i in range(0, N, batch_size):
        i_end = min(i + batch_size, N)
        chunk_i = flat_data[i:i_end].to(device)
        for j in range(0, N, batch_size):
            j_end = min(j + batch_size, N)
            chunk_j = flat_data[j:j_end].to(device)
            chunk_affinity = chunk_i @ chunk_j.T  # unnormalized
            affinity_matrix[i:i_end, j:j_end] = chunk_affinity
            del chunk_j
        del chunk_i
    torch.cuda.empty_cache()
    return affinity_matrix


def symmetrize_matrix_inplace(matrix, chunk_size=1024):
    N = matrix.shape[0]
    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        for j in range(0, N, chunk_size):
            j_end = min(j + chunk_size, N)
            block = matrix[i:i_end, j:j_end]
            transpose_block = matrix[j:j_end, i:i_end].T
            avg_block = (block + transpose_block) / 2
            matrix[i:i_end, j:j_end] = avg_block
            matrix[j:j_end, i:i_end] = avg_block.T
    return matrix


def visualize_affinity_slices_grid(affinity_matrix, probe_pts, save_path=None, res=256):
    """
    Visualize affinity slices at each probe point in a grid.
    """
    num_points = probe_pts.shape[0]
    affinity_matrix = affinity_matrix.reshape(res, res, res, res)

    n_cols = min(num_points, 5)
    n_rows = (num_points + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if n_rows == 1:
        axs = np.expand_dims(axs, 0)
    if n_cols == 1:
        axs = np.expand_dims(axs, 1)

    for idx, (x_probe, y_probe) in enumerate(probe_pts.astype(int)):
        affinity_slice = affinity_matrix[y_probe, x_probe]  # (256, 256)
        row, col = divmod(idx, n_cols)
        ax = axs[row][col]
        ax.imshow(affinity_slice, cmap='hot')
        ax.scatter(x_probe, y_probe, color='black', s=20)
        ax.set_title(f'Object {idx} ({x_probe}, {y_probe})')
        ax.axis('off')

    # Turn off any unused subplots
    for j in range(idx + 1, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axs[row][col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def normalize_affinity(slice_):
    return (slice_ - slice_.min()) / (slice_.max() - slice_.min() + 1e-8)

from ccwm.utils.segment import threshold_heatmap
def get_segment_from_affinity(slice_, thresh=0.5, min_size=50):

    if thresh is None:
        mask = threshold_heatmap(slice_)
    else:
        norm_slice = normalize_affinity(slice_)
        mask = norm_slice > thresh
    # labeled = label(mask)
    # cleaned = remove_small_objects(labeled, min_size=min_size)
    return mask, slice_


def plot_probe_points_on_image(image, probe_pts, color='red', size=30, alpha=0.8):
    """
    Overlay probe points on an image.

    Args:
        image: (H, W, 3) or (H, W) numpy array
        probe_pts: (N, 2) array with [x, y] coordinates
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
    for pt in probe_pts:
        x, y = pt
        plt.scatter(x, y, c=color, s=size, alpha=alpha, edgecolors='black', linewidths=0.5)
    plt.title("Probe Points Overlay")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def mask_nms(segments, iou_thresh=0.5):
    if len(segments) == 0:
        return []

    # Sort by mask area (descending) – keep larger masks first
    # segments = sorted(segments, key=lambda m: -m.sum())
    keep = []

    for i, seg in enumerate(segments):
        suppress = False
        for kept_seg in keep:
            iou = compute_iou(seg, kept_seg)
            if iou > iou_thresh:
                suppress = True
                break
        if not suppress:
            keep.append(seg)

    return np.stack(keep) if keep else np.zeros((0, segments[0].shape[0], segments[0].shape[1]), dtype=np.uint8)


def iterative_affinity_segmentation(affinity_matrix, probe_pts, res=128, thresh=0.0, min_size=50):
    H, W = res, res
    covered = np.zeros(len(probe_pts), dtype=bool)
    affinity_matrix = affinity_matrix.reshape(H, W, H, W)
    all_segments = []
    remaining_indices = np.arange(len(probe_pts))

    while not np.all(covered):
        # Step 1: compute normalized affinity slices for all uncovered probes
        norm_scores = []
        slices = []

        for i in remaining_indices:
            x_probe, y_probe = map(int, probe_pts[i])
            # if res == 128:
            #     x_probe, y_probe = map(lambda v: v // 2, (x_probe, y_probe))
            #
            slice_ = affinity_matrix[y_probe, x_probe]  # (H, W)
            norm = np.linalg.norm(slice_)
            slices.append(slice_)
            norm_scores.append(norm)

        # Step 2: pick the probe with max norm
        best_idx = remaining_indices[np.argmax(norm_scores)]
        x_probe, y_probe = map(int, probe_pts[best_idx])
        base_slice = affinity_matrix[y_probe, x_probe]

        # Step 3: initial segment
        mask, _ = get_segment_from_affinity(base_slice, thresh=thresh, min_size=min_size)

        # Step 4: find all probes inside the segment
        included = []
        for i in remaining_indices:
            x, y = map(int, probe_pts[i])
            if mask[y, x]:
                included.append(i)

        # print(best_idx, included)

        included = np.array(included)
        if len(included) == 0:
            # fail-safe: mark the best one and continue
            included = [best_idx]

        # Step 5: average affinity slices of included points
        avg_affinity = np.mean(
            [affinity_matrix[int(probe_pts[i][1]), int(probe_pts[i][0])] for i in included],
            axis=0
        )

        # Step 6: refine segment with averaged slice
        refined_mask, refined_norm = get_segment_from_affinity(avg_affinity, thresh=None, min_size=min_size)
        all_segments.append(refined_mask)

        # Step 7: mark included probes as covered
        covered[included] = True
        remaining_indices = np.where(~covered)[0]

    return all_segments


def sample_diverse_high_prob_points(prob_map, num_samples=16, min_prob=0.001, min_dist=4):
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


def plot_segment_masks_row(all_segments, figsize_per_mask=3):
    """
    Plot all binary segment masks in a single row.

    Args:
        all_segments: (N, H, W) array of boolean or binary masks
        figsize_per_mask: controls width of each subplot (default: 3)
    """
    N = all_segments.shape[0]
    fig, axs = plt.subplots(1, N, figsize=(figsize_per_mask * N, figsize_per_mask))

    if N == 1:
        axs = [axs]  # ensure iterable if only 1

    for i in range(N):
        ax = axs[i]
        ax.imshow(all_segments[i], cmap='gray')
        ax.set_title(f'Segment {i}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_contour_overlay(ax, mask, color='black'):
    from skimage import measure
    contours = measure.find_contours(mask.astype(float), 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color=color)

def overlay_segment_on_image(image, seg_mask, color=[1.0, 0.0, 0.0], alpha=0.7):
    overlay_image = image.copy()
    seg_mask = seg_mask.astype(bool)
    overlay_image[seg_mask] = (
        alpha * np.array(color) + (1 - alpha) * overlay_image[seg_mask]
    )
    return overlay_image

def plot_segments_grid(image, pm, pm_min, gt_segments, pred_segments, assignment, max_remain_per_row=5, save_path=None):
    fig_rows = 3
    fig_cols = max(len(gt_segments), len(pred_segments), max_remain_per_row)
    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=(3 * fig_cols, 3 * fig_rows))

    if fig_rows == 1:
        axs = axs[np.newaxis, :]
    if fig_cols == 1:
        axs = axs[:, np.newaxis]

    # Row 0: Input + GT
    axs[0, 0].imshow(image)
    axs[0, 0].set_title("Input Image")
    axs[0, 0].axis('off')
    for i, gt_seg in enumerate(gt_segments):
        if i + 1 >= fig_cols:
            break
        overlaid = overlay_segment_on_image(image/255, gt_seg, color=[1.0, 1.0, 0.0])
        axs[0, i + 1].imshow(overlaid)
        plot_contour_overlay(axs[0, i + 1], gt_seg)
        axs[0, i + 1].set_title(f"GT Segment {i}")
        axs[0, i + 1].axis('off')

    # Row 1: Matched Pred Segments
    for i, (gt_idx, pred_idx) in enumerate(zip(assignment[0], assignment[1])):
        if i + 1 >= fig_cols:
            break
        seg = pred_segments[pred_idx]
        overlaid = overlay_segment_on_image(image/255, seg, color=[1.0, 1.0, 0.0])  # green
        axs[1, i+1].imshow(overlaid)
        plot_contour_overlay(axs[1, i+1], seg)
        axs[1, i+1].set_title(f"Matched segment for GT {gt_idx}")
        axs[1, i+1].axis('off')

    # Row 2: Remaining unassigned segments
    assigned_pred_indices = set(assignment[1])
    remaining_pred_indices = [i for i in range(len(pred_segments)) if i not in assigned_pred_indices]

    for i in range(fig_cols):
        if i < len(remaining_pred_indices) and i < max_remain_per_row:
            pred_idx = remaining_pred_indices[i]
            seg = pred_segments[pred_idx]
            overlaid = overlay_segment_on_image(image/255, seg, color=[1.0, 1.0, 0.0])  # blue
            axs[2, i+1].imshow(overlaid)
            plot_contour_overlay(axs[2, i+1], seg)
            axs[2, i+1].set_title(f"Unmatched Segment")
        axs[2, i].axis('off')
        axs[1, i].axis('off')
        axs[0, i].axis('off')

    # Heatmap subplot
    im1 = axs[1, 0].imshow(pm, cmap='hot')
    axs[1, 0].set_title("Motion Probability")
    axs[1, 0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im1 = axs[2, 0].imshow(pm_min, cmap='hot')
    axs[2, 0].set_title("Min Pooled")
    axs[2, 0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axs[2, 0], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    if save_path is not None:
        plt.close()
    else:
        plt.show()



def evaluate_AP_AR_single_image(pred_segments, gt_segments):
    """
    Compute Average Precision (AP) and Average Recall (AR) for a single image.
    Precision and Recall are computed over IoU=.50:.05:.95.

    The procedure is as follows:
      1. Assign predicted segments to one of the gt segments, such that
        the IoU between gt and pred in each bin is maximized (globally).
      2. Compute True Positives, i.e. number of bins such that
        the IoU between gt and pred is greater than IoU threshold.
      3. Compute Precision and Recall, average across IoU thresholds.

    Arguments:
      pred_segments (N, H, W): N predicted segment masks.
      gt_segments (M, H, W): M ground truth segment masks.

    Returns:
      (dict(str: any)): A dictionary containing evaluation results.
    """

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