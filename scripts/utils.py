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
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# took this from SpelkeBench implementation!
def offset_multiple_centroids(centroids, N, min_mag=10.0, max_mag=25.0):
    """
    Applies N offset vectors to each centroid in centroids, using same directions.

    centroids: Tensor of shape (M, 2) in (y, x) format
    Returns: Tensor of shape (N, M, 4) as [y1, x1, y2, x2]
    """
    device = centroids.device
    M = centroids.shape[0]

    # Angles and directions: shape (N,)
    angles = torch.arange(N, device=device) * (2 * math.pi / N)

    dx_unit = torch.cos(angles)  # (N,)
    dy_unit = torch.sin(angles)  # (N,)

    # Sample one magnitude per direction (shared across all centroids)
    magnitudes = torch.rand(N, device=device) * (max_mag - min_mag) + min_mag  # (N,)
    dx = magnitudes * dx_unit  # (N,)
    dy = magnitudes * dy_unit  # (N,)

    return dx, dy

def pixel_to_normalized(coords: np.ndarray, img_shape: Tuple[int, int]) -> torch.Tensor:
    """Convert pixel to [0, 1] normalized coordinates for model"""
    h, w = img_shape
    coords_tensor = torch.tensor(coords, dtype=torch.float32)

    normalized = coords_tensor.clone()
    normalized[..., 0] = coords_tensor[..., 0] / w  # x
    normalized[..., 1] = coords_tensor[..., 1] / h  # y

    return normalized


def normalized_to_pixel(coords: torch.Tensor, img_shape: Tuple[int, int]) -> np.ndarray:
    """Convert [0, 1] coordinates to pixel coordinates for vis"""
    h, w = img_shape
    coords_np = coords.cpu().numpy()

    pixel_coords = np.zeros_like(coords_np)
    pixel_coords[..., 0] = coords_np[..., 0] * w  # x
    pixel_coords[..., 1] = coords_np[..., 1] * h  # y

    return pixel_coords

def get_dot_product_map(avg_flow, flow_cond_with_obj):
    # Compute flow direction vector
    dx = flow_cond_with_obj[2] - flow_cond_with_obj[0]
    dy = flow_cond_with_obj[3] - flow_cond_with_obj[1]

    # dx, dy = np.array(flow_cond_with_obj[-1][2:]) - np.array(flow_cond_with_obj[-1][0:2])
    direction = np.array([dx, dy])

    # Compute dot product between avg_flow and direction vector
    dot_prod = torch.sum(
        avg_flow * torch.tensor(direction, dtype=avg_flow.dtype, device=avg_flow.device)[None, None, :], dim=-1)
    # dot_prod_np = #dot_prod.cpu().numpy()

    return dot_prod

def threshold_heatmap(heatmap):
    # Step 1: Min-Max Normalize to [0, 255]
    heatmap_min = np.min(heatmap)
    heatmap_max = np.max(heatmap)
    heatmap_norm = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)  # Avoid divide-by-zero
    heatmap_scaled = (heatmap_norm * 255).astype(np.uint8)

    # Step 2: Apply Otsu thresholding
    _, thresh = cv2.threshold(heatmap_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh // 255

def reconstruct_mixture_distribution(chunks, device): # for kl reconstructing from batched chunks
    # concat chunks
    mixture_probs = torch.cat([chunk['mixture_probs'] for chunk in chunks], dim=1).to(device)
    component_loc = torch.cat([chunk['component_loc'] for chunk in chunks], dim=1).to(device)
    component_cov = torch.cat([chunk['component_cov'] for chunk in chunks], dim=1).to(device)

    # distributions
    mixture_dist = Categorical(probs=mixture_probs)
    component_dist = MultivariateNormal(loc=component_loc, covariance_matrix=component_cov)

    return MixtureSameFamily(mixture_dist, component_dist)