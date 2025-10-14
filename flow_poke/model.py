# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025 Stefan Baumann et al., CompVis @ LMU Munich

import math
from typing import Literal, Any
from functools import reduce, partial
from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask, create_block_mask
from torch.distributions.multivariate_normal import MultivariateNormal as _MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily as _MixtureSameFamily
from torch.distributions.categorical import Categorical as _Categorical
from einops import rearrange, repeat
from jaxtyping import Float, Bool
from tqdm.auto import trange

from .dinov2 import DinoFeatureExtractor


# ---------------------------------------------------------------------------------------------------------------------
# Modules & Utilities
# ---------------------------------------------------------------------------------------------------------------------


class AdaRMSNorm(nn.Module):
    def __init__(self, features: int, cond_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.features = (features,)
        self.linear = nn.Linear(cond_features, features, bias=False)
        torch.nn.init.zeros_(self.linear.weight)

    def forward(
        self, x: Float[torch.Tensor, "... l d"], cond: Float[torch.Tensor, "... d_cond"]
    ) -> Float[torch.Tensor, "... l d"]:
        return F.rms_norm(x, self.features, weight=None, eps=self.eps) * (1 + self.linear(cond)[..., None, :])


class ResidualWrapper(nn.Sequential):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + x


def squareplus(x: torch.Tensor) -> torch.Tensor:
    # See https://x.com/jon_barron/status/1387167648669048833
    return (x + torch.sqrt(x**2 + 4)) / 2


# ---------------------------------------------------------------------------------------------------------------------
# Attention & RoPE Utilities
# Mostly mirrors HDiT (Crowson et al., ICML 2024)
# Importantly, the coordinate order is flipped compared to HDiT: we index using (x, y) aligning with common conventions
# Additionally, we map images to the [0, 1]^2 range, as opposed to [-1, 1]^2 to make the flow values more intuitive
# Specifically, (0, 0) is the top left corner, (1, 0) is the top right corner, (0, 1) is the bottom left corner
# We retain the exact RoPE behavior from HDiT by scaling back to [-1, 1] in the RoPE theta computation
# ---------------------------------------------------------------------------------------------------------------------


def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype).abs() + eps)  # https://github.com/crowsonkb/k-diffusion/issues/111
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    # assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)


class AxialRoPE2D(nn.Module):
    def __init__(self, d_head: int, n_heads: int):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads

        min_freq = math.pi
        max_freq = 10.0 * math.pi

        log_min = math.log(min_freq)
        log_max = math.log(max_freq)
        freqs = torch.stack([torch.linspace(log_min, log_max, n_heads * d_head // 4 + 1)[:-1].exp()] * 2)
        self.freqs = nn.Parameter(freqs.view(2, d_head // 4, n_heads).mT.contiguous(), requires_grad=False)

    def forward(self, pos: Float[torch.Tensor, "b l 2"]) -> Float[torch.Tensor, "b n_h l d_head"]:
        # Undo the coordinate system convention change and go back to [-1, 1]^2 by using pos * 2 - 1
        theta_w = pos[..., None, 0:1].mul(2).sub(1) * self.freqs[0].to(pos.dtype)
        theta_h = pos[..., None, 1:2].mul(2).sub(1) * self.freqs[1].to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1).movedim(-2, -3)


def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def make_grid(h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1)
    h, w, d = grid.shape
    return grid.view(h * w, d)


def bounding_box(h, w, pixel_aspect_ratio=1.0):
    # Adjusted dimensions
    w_adj = w
    h_adj = h * pixel_aspect_ratio

    # Adjusted aspect ratio
    ar_adj = w_adj / h_adj

    # Determine bounding box based on the adjusted aspect ratio
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj

    return y_min, y_max, x_min, x_max


def make_axial_pos_2d(
    h: int, w: int, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None, relative_pos=True
) -> Float[torch.Tensor, "(h w) 2"]:
    if relative_pos:
        y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    else:
        y_min, y_max, x_min, x_max = -h / 2, h / 2, -w / 2, w / 2

    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    # Change coordinate system convention from (y, x), [-1, 1]^2 to (x, y), [0, 1]^2
    return make_grid(h_pos, w_pos).flip(-1).mul(0.5).add(0.5)


# ---------------------------------------------------------------------------------------------------------------------
# Attention Masks
# ---------------------------------------------------------------------------------------------------------------------


def query_causal_mask_mod(sequence_length: int, n_query: int = 1):
    """
    Seq: [pokes, queries], len(pokes) = sequence_length, len(queries) = n_query * sequence_length
    n_query queries attend to the same set of pokes each, not to each other
    Pokes attend causally to each other, but not to queries
    First query set attends to no pokes, next set to the first poke, then first two pokes, etc
    """

    def mask_mod(batch, head, q_idx, kv_idx):
        return (
            ((q_idx >= kv_idx) & (q_idx < sequence_length))  # Normal causal part
            | (q_idx == kv_idx)  # Diagonal
            | (
                ((q_idx - sequence_length - n_query) // n_query >= kv_idx)
                & (q_idx >= sequence_length)
                & (kv_idx < sequence_length)
            )  # Query heads
        )

    return mask_mod


def inference_query_causal_mask_mod(sequence_length: int):
    """
    Seq: [pokes, queries], len(pokes) = sequence_length, len(queries) = *
    Pokes attend causally to each other, but not to queries
    Queries attend to all pokes, but not to each other
    """

    def mask_mod(batch, head, q_idx, kv_idx):
        return (
            (q_idx == kv_idx)  # Diagonal
            | ((q_idx >= kv_idx) & (q_idx < sequence_length))  # Normal causal part
            | ((kv_idx < sequence_length) & (q_idx >= sequence_length))  # Queries
        )

    return mask_mod


# ---------------------------------------------------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------------------------------------------------


class KVCache(nn.Module):
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.register_buffer("k", torch.zeros(shape, dtype=dtype, device=device), persistent=False)
        self.register_buffer("v", torch.zeros(shape, dtype=dtype, device=device), persistent=False)


# ---------------------------------------------------------------------------------------------------------------------
# Main Transformer
# The transformer is, for the most part, a modern standard LLama2-style transformer,
# with some further modifications primarily inspired by HDiT (Crowson et al., ICML 2024)
# Main differences to a "standard" transformer:
# - Cross-attention for image features
# - No biases for linear layers
# - LayerNorm -> RMSNorm
# - GELU -> SwiGLU
# - Additive PE -> RoPE, using the HDiT approach to implement 2D RoPE
# - d_k-scaled attention -> cossim-attention with learnable scales
# - Adaptive norms to introduce conditioning
# Notable somewhat specific implementation details for FPT:
# - Cross-attention gets full RoPE-based PE
# - We use custom attention masks for self-attention, which are implemented via flex attention
#   Sometimes during inference, we might have strongly varying (but not particularly sparse) attention patterns,
#   for which case we also have optional fallbacks to full causal attention (mask="causal") or a boolean mask tensor
# ---------------------------------------------------------------------------------------------------------------------


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_cross: int,
        d_head: int = 64,
        dropout: float = 0.0,
        ff_expand: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = d_model // d_head

        self.self_norm = AdaRMSNorm(d_model, cond_features=d_model)
        self.cross_norm = AdaRMSNorm(d_model, cond_features=d_model)
        self.ffn_norm = AdaRMSNorm(d_model, cond_features=d_model)

        self.self_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.self_out = nn.Linear(d_model, d_model, bias=False)
        self.cross_q = nn.Linear(d_model, d_model, bias=False)
        self.cross_kv = nn.Linear(d_cross, d_model * 2, bias=False)
        self.cross_out = nn.Linear(d_model, d_model, bias=False)
        self.ffn_up = nn.Linear(d_model, 2 * ff_expand * d_model, bias=False)
        self.ffn_down = nn.Linear(ff_expand * d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.self_scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.cross_scale = nn.Parameter(torch.full([self.n_heads], 10.0))

        # Init
        nn.init.zeros_(self.ffn_down.weight)
        nn.init.zeros_(self.self_out.weight)
        nn.init.zeros_(self.cross_out.weight)

        # Zero-length KV cache by default, only used during AR inference
        self.kv_cache = KVCache((1, self.n_heads, 0, self.d_head), dtype=torch.bfloat16, device=None)

    def fwd_self(
        self,
        x: Float[torch.Tensor, "b l d"],
        theta,
        mask: BlockMask | Literal["causal"] | Bool[torch.Tensor, "1 1 l_q l_kv"],
        i_kv: int | None = None,
    ):
        q, k, v = rearrange(self.self_qkv(x), "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.self_scale[:, None, None], 1e-6)
        q, k = apply_rotary_emb(q, theta), apply_rotary_emb(k, theta)

        # If KV caching is enabled, write the current KV into the cache and then use the cache
        # For typical AR transformers without a query token, we'd only add one token per step, which we subsequently keep
        # In our case, we'll have two previously unseen tokens per step (poke for last sample, query for next),
        # only one of which we're gonna keep; but we can just write both to the cache as the query will be overwritten
        if i_kv is not None:
            L = k.size(2)
            # print(f"{k.shape=}, {self.kv_cache.k[:, :, i_kv : i_kv + L].shape=}")
            self.kv_cache.k[:, :, i_kv : i_kv + L] = k
            self.kv_cache.v[:, :, i_kv : i_kv + L] = v
            k = self.kv_cache.k[:, :, : i_kv + L]
            v = self.kv_cache.v[:, :, : i_kv + L]

        if isinstance(mask, str) and mask == "causal":
            x = F.scaled_dot_product_attention(q, k, v, scale=1.0, is_causal=True)
        elif isinstance(mask, torch.Tensor):
            x = F.scaled_dot_product_attention(q, k, v, scale=1.0, attn_mask=mask)
        else:
            x = torch.compile(flex_attention)(q, k, v, scale=1.0, block_mask=mask)  # type: ignore

        return self.self_out(self.dropout(rearrange(x, "n nh l e -> n l (nh e)")))

    def fwd_cross(self, x, x_cross, theta, theta_cross):
        q = rearrange(self.cross_q(x), "n l (nh e) -> n nh l e", e=self.d_head)
        k, v = rearrange(self.cross_kv(x_cross), "n l (t nh e) -> t n nh l e", t=2, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.cross_scale[:, None, None], 1e-6)
        q, k = apply_rotary_emb(q, theta), apply_rotary_emb(k, theta_cross)
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        return rearrange(x, "n nh l e -> n l (nh e)")

    def fwd_ffn(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        x, gate = self.ffn_up(x).chunk(2, dim=-1)
        return self.ffn_down(self.dropout(x * F.silu(gate)))

    def forward(
        self,
        x: Float[torch.Tensor, "b l d_model"],
        x_cross: Float[torch.Tensor, "b l' d_cross"],
        theta: Float[torch.Tensor, "b n_h l d_head"],
        theta_cross: Float[torch.Tensor, "b n_h l' d_head"],
        cond: Float[torch.Tensor, "b d_model"],
        mask: BlockMask | Literal["causal"] | Bool[torch.Tensor, "1 1 l_q l_kv"],
        i_kv: int | None = None,
    ):
        x = x + self.fwd_self(self.self_norm(x, cond=cond), theta=theta, mask=mask, i_kv=i_kv)
        x = x + self.fwd_cross(self.cross_norm(x, cond=cond), x_cross=x_cross, theta=theta, theta_cross=theta_cross)
        x = x + self.fwd_ffn(self.ffn_norm(x, cond=cond))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, depth: int, d_cross: int, layer_params: dict[str, Any] = {}):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model=width, d_cross=d_cross, **layer_params) for _ in range(depth)]
        )
        d_head = layer_params.get("d_head", 64)
        n_heads = width // d_head
        self.pos_emb = AxialRoPE2D(d_head=d_head, n_heads=n_heads)

    def forward(
        self,
        x: Float[torch.Tensor, "b l d_in"],
        pos: Float[torch.Tensor, "b l 2"],
        x_cross: Float[torch.Tensor, "b l_c d_cross"],
        pos_cross: Float[torch.Tensor, "b l_c 2"],
        **kwargs,
    ) -> Float[torch.Tensor, "b l d_out"]:
        theta, theta_cross = self.pos_emb(pos), self.pos_emb(pos_cross)
        for layer in self.layers:
            x = layer(x, x_cross=x_cross, theta=theta, theta_cross=theta_cross, **kwargs)
        return x

    def reset_kv_cache(self, b: int, l: int):
        for layer in self.layers:
            layer.kv_cache.k = layer.kv_cache.k.new_zeros((b, layer.kv_cache.k.size(1), l, layer.kv_cache.k.size(3)))
            layer.kv_cache.v = layer.kv_cache.v.new_zeros((b, layer.kv_cache.v.size(1), l, layer.kv_cache.v.size(3)))

    def grow_kv_cache(self, b: int, l: int):
        for layer in self.layers:
            b_current, _, l_current, _ = layer.kv_cache.k.shape
            if b_current < b or l_current < l:
                print(f"Growing KV cache from {b_current}x{l_current} to {b}x{l}", flush=True)
                layer.kv_cache.k = layer.kv_cache.k.new_zeros(
                    (b, layer.kv_cache.k.size(1), l, layer.kv_cache.k.size(3))
                )
                layer.kv_cache.v = layer.kv_cache.v.new_zeros(
                    (b, layer.kv_cache.v.size(1), l, layer.kv_cache.v.size(3))
                )


# ---------------------------------------------------------------------------------------------------------------------
# Slicable Distributions
# Slim wrappers around torch.distributions that allow for slicing
# These should work fine for any slicing on leading dimensions, but may not behave as expected for trailing dimensions
# No clue why torch.distributions doesn't support this out of the box, seems like a very basic feature
# We also include a numerical approximation for the cdf of a bivariate normal via a Gauss-Legendre quadrature
# TODO: consider moving to a separate file
# ---------------------------------------------------------------------------------------------------------------------


class MultivariateNormal(_MultivariateNormal):
    def __getitem__(self, idx) -> "MultivariateNormal":
        idx = idx if isinstance(idx, tuple) else (idx,)
        # Equivalent of MultivariateNormal(self.loc[idx, :], scale_tril=self.scale_tril[idx, :, :])
        return MultivariateNormal(
            self.loc[idx + (slice(None),)],
            scale_tril=self.scale_tril[idx + (slice(None), slice(None))],  # type: ignore
        )

    @staticmethod
    def _gauss_legendre_nodes_weights(n: int, device, dtype):
        k = torch.arange(1, n, device=device, dtype=dtype)
        beta = k / torch.sqrt(4 * k * k - 1)

        J = torch.zeros((n, n), device=device, dtype=dtype)
        J.diagonal(1).copy_(beta)
        J.diagonal(-1).copy_(beta)

        eigvals, eigvecs = torch.linalg.eigh(J)
        x = eigvals
        w = 2 * (eigvecs[0, :] ** 2)
        return x, w

    @staticmethod
    def _bvn_cdf(a, b, rho, n_nodes: int):
        device = a.device
        dtype = torch.float64
        a = a.to(dtype)
        b = b.to(dtype)
        rho = rho.to(dtype)

        stdn = torch.distributions.Normal(
            loc=torch.tensor(0.0, device=device, dtype=dtype), scale=torch.tensor(1.0, device=device, dtype=dtype)
        )
        base = stdn.cdf(a) * stdn.cdf(b)

        r = rho.abs().clamp_max(1.0 - 1e-12)
        small = r < 1e-12
        if small.all():
            return base

        x, w = MultivariateNormal._gauss_legendre_nodes_weights(n_nodes, device=device, dtype=dtype)

        bro_shape = torch.broadcast_shapes(a.shape, b.shape, r.shape)
        expand_shape = (n_nodes,) + (1,) * len(bro_shape)
        xN = x.view(*expand_shape)
        wN = w.view(*expand_shape)

        s = torch.sign(rho)
        t = (0.5 * r).unsqueeze(0) * (xN + 1.0)
        t_signed = s.unsqueeze(0) * t

        one = torch.tensor(1.0, device=device, dtype=dtype)
        denom = torch.sqrt(one - t_signed**2)
        num = torch.exp(
            -(a.unsqueeze(0) ** 2 - 2 * t_signed * a.unsqueeze(0) * b.unsqueeze(0) + b.unsqueeze(0) ** 2)
            / (2 * (one - t_signed**2))
        )
        integrand = num / (2 * torch.pi * denom)

        I = (0.5 * r).unsqueeze(0) * (wN * integrand).sum(dim=0)
        out = base + s * I
        out = torch.where(small, base, out)
        return out

    def cdf(self, value, n_nodes: int = 32):
        if value.shape[-1] != 2:
            raise NotImplementedError("CDF approximation is only implemented for bivariate normal.")

        dtype_work = torch.float64
        value = value.to(dtype_work)
        mu = self.loc.to(dtype_work)
        Sigma = self.covariance_matrix.to(dtype_work)

        var_x = Sigma[..., 0, 0].clamp_min(0)
        var_y = Sigma[..., 1, 1].clamp_min(0)
        sig_x = torch.sqrt(var_x + 0.0)
        sig_y = torch.sqrt(var_y + 0.0)
        rho = (Sigma[..., 0, 1] / (sig_x * sig_y)).clamp(-1 + 1e-12, 1 - 1e-12)

        a = (value[..., 0] - mu[..., 0]) / sig_x
        b = (value[..., 1] - mu[..., 1]) / sig_y

        F = MultivariateNormal._bvn_cdf(a, b, rho, n_nodes=n_nodes)
        return F.to(self.loc.dtype)


class Categorical(_Categorical):
    def __getitem__(self, idx) -> "Categorical":
        idx = idx if isinstance(idx, tuple) else (idx,)
        # Equivalent of Categorical(logits=self.logits[idx, :])
        return Categorical(logits=self.logits[idx + (slice(None),)])  # type: ignore


class MixtureSameFamily(_MixtureSameFamily):
    def __getitem__(self, idx) -> "MixtureSameFamily":
        idx = idx if isinstance(idx, tuple) else (idx,)
        # Equivalent of MixtureSameFamily(self.mixture_distribution[idx], self.component_distribution[idx, :])
        return MixtureSameFamily(self.mixture_distribution[idx], self.component_distribution[idx + (slice(None),)])


# ---------------------------------------------------------------------------------------------------------------------
# Flow Poke Transformer Input & Output MLPs
# The input MLP jointly embeds given flow (for pokes, for queries, this is replaced by a learnable token) and image features
# The flow embedding is a Fourier embedding with fixed coefficients, with the flow constrained to [-1/prescale, 1/prescale]
# using a tanh function to prevent wraparound issues with unexpectedly large flow values. The default range is [-2, 2]
# The image features are the same as are cross-attended to, just retrieved for each position using bilinear sampling
# The output MLP is just a simple MLP with an RMSNorm at the input
# ---------------------------------------------------------------------------------------------------------------------


class FlowPokeInputMLP(nn.Module):
    def __init__(self, width: int, d_aux: int, flow_dim: int = 2, depth: int = 2, flow_prescale: float = 0.5):
        super().__init__()
        self.flow_dim = flow_dim
        self.flow_prescale = flow_prescale
        self.mlp = nn.Sequential(
            nn.Linear(self.flow_dim * width + d_aux, width, bias=False),
            *[
                ResidualWrapper(
                    nn.RMSNorm(width),
                    nn.Linear(width, width, bias=False),
                    nn.SiLU(),
                    nn.Linear(width, width, bias=False),
                )
                for _ in range(depth)
            ],
        )
        self.query_token = nn.Parameter(torch.empty(1, 1, self.flow_dim * width), requires_grad=True)
        torch.nn.init.trunc_normal_(self.query_token)
        self.register_buffer("coeffs", (2 * torch.pi * (torch.arange(width // 2) + 1))[None, None, None, :])

    def forward(
        self,
        pos: Float[torch.Tensor, "b l 2"],
        flow: Float[torch.Tensor, "b l d_flow"],
        is_query: Bool[torch.Tensor, "b l"],
        aux_feats: Float[torch.Tensor, "b d_aux h w"],
    ) -> Float[torch.Tensor, "b l d_model"]:
        B, L, _ = pos.shape
        # map approximate [-1/prescale, 1/prescale] range to [0, 1]
        flow_scaled: Float[torch.Tensor, "b l d_flow"] = flow.mul(self.flow_prescale).tanh().add(1).div(2)
        flow_: Float[torch.Tensor, "b l d_flow n_coeff"] = flow_scaled[..., None] * self.coeffs.to(flow)  # type: ignore
        flow_fourier: Float[torch.Tensor, "b l (d_flow d_model)"] = rearrange(
            torch.stack([torch.sin(flow_), torch.cos(flow_)]), "sc b l d_flow n_coeff -> b l (d_flow sc n_coeff)"
        )
        # Mask flow embedding for queries with learnable token
        flow_emb = torch.where(is_query[..., None], self.query_token, flow_fourier)
        # Retrieve image features for each position
        aux_feats_: Float[torch.Tensor, "b l d_aux"] = rearrange(
            F.grid_sample(
                aux_feats,
                pos[:, :, None, :].mul(2).sub(1),  # grid_sample expects [-1, 1]^2, pos is in [0, 1]^2
                align_corners=True,
                mode="bilinear",
                padding_mode="border",
            ),
            "b d_aux l 1 -> b l d_aux",
        )
        # Concatenate & jointly embed flow embeddings and image features
        return self.mlp(torch.cat([flow_emb, aux_feats_], dim=-1))


class FlowPokeOutputMLP(nn.Sequential):
    def __init__(self, width: int, d_out: int, depth: int = 2):
        super().__init__(
            nn.RMSNorm(width, eps=1e-6),
            *[l for _ in range(depth) for l in [nn.Linear(width, width), nn.SiLU()]],
            nn.Linear(width, d_out),
        )


# ---------------------------------------------------------------------------------------------------------------------
# Main Flow Poke Transformer
# The FPT class wraps the inner transformer and adds the various interesting bits around it
# - Image embedder: embeds the image. Initialized to DINOv2-R, but unlocked during training
# - Input embedding: embeds the fourier-embedded flow (for pokes) and retrieved image features using an MLP
# - Transformer: fairly straight-forward transformer, see above for details
# - GMM head: MLP, followed by mapping to pointwise GMMs. Important: uses fp64 precision for the GMM parameters
#     - The parameters (locs, scales, logits) are derived from the post-output MLP tensor
#     - The scales (covariance matrices) are predicted as lower-triangular Cholesky factors, where we softclip the
#       diagonal to be greater than zero to ensure positive-definiteness
# The image embedding step has deliberately been separated out from the main forward, as it can be cached in many cases
# ---------------------------------------------------------------------------------------------------------------------


class FlowPokeTransformer(nn.Module):
    def __init__(
        self,
        # Transformer parameters
        width: int,
        depth: int,
        layer_params: dict[str, Any] = {},
        # Auxiliary module parameters
        img_embedder_params: dict[str, Any] = {},
        input_mlp_params: dict[str, Any] = {},
        output_mlp_params: dict[str, Any] = {},
        # GMM head parameters
        flow_dim: int = 2,
        n_components: int = 4,
        dist_dtype: torch.dtype = torch.float64,
        # Other parameters
        train_image_feature_extractor: bool = True,
    ):
        super().__init__()

        # Input stuff
        self.train_image_feature_extractor = train_image_feature_extractor
        self.image_embedder = DinoFeatureExtractor(requires_grad=train_image_feature_extractor, **img_embedder_params)
        d_img_feat = self.image_embedder.embed_dim
        self.cond_emb = nn.Embedding(2, width)
        self.input_mlp = FlowPokeInputMLP(width, d_aux=d_img_feat, flow_dim=flow_dim, **input_mlp_params)

        # Inner transformer
        self.transformer = Transformer(width, depth, d_img_feat, layer_params=layer_params)

        # Output stuff
        self.n_components = n_components
        self.flow_dim = flow_dim
        self.out_mlp = FlowPokeOutputMLP(
            width, n_components * (1 + flow_dim + (flow_dim * (flow_dim + 1)) // 2), **output_mlp_params
        )
        self.dist_dtype = dist_dtype

    def embed_image(self, img: Float[torch.Tensor, "b c h w"]) -> dict[str, torch.Tensor]:
        """
        Get the image features, pass to forward() etc. Image is expected to be square and in [-1, 1].
        If the resolution is not the expected one, it will be resized automatically inside the image embedder.
        """
        with nullcontext() if self.train_image_feature_extractor else torch.no_grad():
            features: Float[torch.Tensor, "b d h w"] = self.image_embedder(img)
            B, D, H, W = features.shape
            feature_pos: Float[torch.Tensor, "(h w) 2"] = make_axial_pos_2d(H, W, device=features.device)
        return {
            "feat": features,
            "pos": repeat(feature_pos, "(h w) d -> b h w d", b=B, h=H, w=W),
        }

    def forward(
        self,
        pos: Float[torch.Tensor, "b l 2"],
        flow: Float[torch.Tensor, "b l d_flow"],
        is_query: Bool[torch.Tensor, "b l"],
        camera_static: Bool[torch.Tensor, "b"],
        mask: BlockMask | Literal["causal"] | Bool[torch.Tensor, "1 1 l_q l_kv"],
        d_img: dict[str, torch.Tensor],
        i_kv: int | None = None,
    ) -> MixtureSameFamily:
        """
        Main forward. x_cross & pos_cross are obtained via embed_image().
        Coordinate system is (x, y), [0, 1]^2.
        Flow is in that same coordinate system, resulting in [-1, 1]^2 for normal usage
        """
        cond = self.cond_emb(camera_static.long())
        x = self.transformer(
            x=self.input_mlp(pos=pos, flow=flow, is_query=is_query, aux_feats=d_img["feat"]),
            pos=pos,
            x_cross=rearrange(d_img["feat"], "b d h w -> b (h w) d"),
            pos_cross=rearrange(d_img["pos"], "b h w d -> b (h w) d"),
            cond=cond,
            mask=mask,
            i_kv=i_kv,
        )
        return self._get_distribution(self.out_mlp(x))

    def _get_distribution(self, x_out: Float[torch.Tensor, "b l d_model"]) -> MixtureSameFamily:
        B, L, _ = x_out.shape
        x = x_out.to(self.dist_dtype)
        logits, x = x[..., : self.n_components], x[..., self.n_components :]
        locs, scale_vals = x[..., : self.n_components * self.flow_dim], x[..., self.n_components * self.flow_dim :]
        locs = rearrange(locs, "b l (n d) -> b l n d", d=self.flow_dim)
        scales_tril = scale_vals.new_zeros((B, L, self.n_components, self.flow_dim, self.flow_dim))
        # Map scale params to lower-triangular matrix
        scales_tril[..., *torch.tril_indices(self.flow_dim, self.flow_dim, device=x.device)] = scale_vals.view(
            B, L, self.n_components, (self.flow_dim * (self.flow_dim + 1)) // 2
        )
        # Softclip diagonal of covariance matrix to be > 0
        diag_mask = torch.eye(self.flow_dim, device=scales_tril.device, dtype=scales_tril.dtype)[None, None, None]
        scales_tril = scales_tril * (1 - diag_mask) + diag_mask * (squareplus(scales_tril) + 1e-4)
        return MixtureSameFamily(
            Categorical(logits=logits),
            MultivariateNormal(
                loc=locs,
                scale_tril=scales_tril,
            ),
        )

    def predict_parallel(
        self,
        poke_pos: Float[torch.Tensor, "b l c"],
        poke_flow: Float[torch.Tensor, "b l c"],
        query_pos: Float[torch.Tensor, "b l_q c"],
        camera_static: Bool[torch.Tensor, "b"] | bool,
        d_img: dict[str, torch.Tensor],
    ) -> MixtureSameFamily:
        """
        Predict motion for one or multiple query points in parallel.
        All queries see all pokes, but are mutually independent.
        """
        B, L_P, C = poke_pos.shape
        L_Q = query_pos.shape[1]
        if isinstance(camera_static, bool):
            camera_static = camera_static * torch.ones(B, device=poke_pos.device, dtype=torch.bool)
        # TODO: validate input shapes

        # We use a proper block mask when predicting multiple queries,
        # but we can do a cheap fall-back to causal sdpa for single queries
        if L_Q <= 1:
            mask = "causal"
        else:
            mask_mod = inference_query_causal_mask_mod(L_P)
            mask = create_block_mask(mask_mod, B=1, H=1, Q_LEN=L_P + L_Q, KV_LEN=L_P + L_Q, device=poke_pos.device)

        return self.forward(
            pos=torch.cat([poke_pos, query_pos], dim=1),
            flow=torch.cat([poke_flow, poke_flow.new_zeros((B, L_Q, C))], dim=1),
            is_query=torch.cat(
                [poke_pos.new_zeros((B, L_P), dtype=torch.bool), poke_pos.new_ones((B, L_Q), dtype=torch.bool)], dim=1
            ),
            camera_static=camera_static,
            mask=mask,
            d_img=d_img,
        )[:, L_P:]

    @torch.no_grad()
    def predict_autoregressive(
        self,
        poke_pos: Float[torch.Tensor, "b l c"],
        poke_flow: Float[torch.Tensor, "b l c"],
        query_pos: Float[torch.Tensor, "b l_q c"],
        camera_static: Bool[torch.Tensor, "b"] | bool,
        d_img: dict[str, torch.Tensor],
        randomize_order: bool = False,
        verbose: bool = True,
    ) -> Float[torch.Tensor, "b l_q c"]:
        """
        Sample motion for multiple (or one) query points in autoregressive manner.
        All queries see all pokes and the sampled motion for preceding queries.

        randomize_order=True is recommended for applications where queries represent a dense grid of points, as
        ordered sequential sampling on a grid is effectively OOD for the model, which has been trained with random orders.
        Anything beyond 128 pokes is also OOD for the model, but it typically works out quite well.
        """
        B, L_P, C = poke_pos.shape
        L_Q = query_pos.shape[1]
        if isinstance(camera_static, bool):
            camera_static = camera_static * torch.ones(B, device=poke_pos.device, dtype=torch.bool)
        # TODO: validate input shapes

        if randomize_order:
            order = torch.stack([torch.randperm(L_Q) for _ in range(B)])
            query_pos = torch.stack([query_pos[i_b, order[i_b]] for i_b in range(B)])

        query_flow = query_pos.new_zeros((B, L_Q, poke_flow.size(-1)))
        # If we were to reset the KV cache every AR generation, we'd be recreating the cache tensor every time,
        # which would break (as of torch ~2.8) the CUDAGraph capture, causing a full re-record
        # So let's just grow the KV cache if needed, but keep it identical otherwise
        self.transformer.grow_kv_cache(b=B, l=(L_P + L_Q))
        # for i_step in range(L_Q):
        for i_step in trange(L_Q, desc="Sampling", disable=not verbose):
            # Unlike normal AR transformers, KV-cached inference becomes a bit more complicated here
            # Primarily, we have the aspect that we have two new tokens per step (poke for last sample, query for next),
            # only one of which we're gonna keep. Normally, one would do full self-attention in KV-cached self-attention,
            # as only one new token is passed, and this is thus equivalent to full causal attention. However, in our case,
            # we have two new tokens, and will need to provide a custom causal mask for that case
            if i_step == 0:
                dist = self.forward(
                    pos=torch.cat([poke_pos, query_pos[:, :1]], dim=1),
                    flow=torch.cat([poke_flow, query_flow[:, :1]], dim=1),
                    is_query=torch.cat(
                        [poke_pos.new_zeros((B, L_P), dtype=torch.bool), poke_pos.new_ones((B, 1), dtype=torch.bool)],
                        dim=1,
                    ),
                    camera_static=camera_static,
                    mask="causal",
                    i_kv=0,
                    d_img=d_img,
                )[:, L_P:]
            else:
                mask = torch.ones((1, 1, 2, L_P + i_step + 1), device=poke_pos.device, dtype=torch.bool)
                mask[:, :, 0, L_P + i_step] = False

                dist = self.forward(
                    pos=query_pos[:, i_step - 1 : i_step + 1],
                    flow=query_flow[:, i_step - 1 : i_step + 1],
                    is_query=torch.cat(
                        [poke_pos.new_zeros((B, 1), dtype=torch.bool), poke_pos.new_ones((B, 1), dtype=torch.bool)],
                        dim=1,
                    ),
                    camera_static=camera_static,
                    mask=mask,
                    i_kv=L_P + i_step - 1,
                    d_img=d_img,
                )[:, 1:]
            query_flow[:, i_step] = dist.sample()

        if randomize_order:
            # Reverse previous random ordering
            order_inv = order.argsort(dim=1)
            query_flow = torch.stack([query_flow[i_b, order_inv[i_b]] for i_b in range(B)])

        return query_flow

    # TODO: transfer over (efficient) moving part KL computation


FlowPokeTransformer_Base = partial(FlowPokeTransformer, width=768, depth=12)
