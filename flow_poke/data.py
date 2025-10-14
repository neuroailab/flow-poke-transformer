# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025 Stefan Baumann et al., CompVis @ LMU Munich

from pathlib import Path
import io
import random
from functools import partial

import torch
import numpy as np
import torchvision.transforms.v2 as TVT
import av
import einops
import webdataset as wds
from jaxtyping import Float, Bool


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True, **kwargs):
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}  # remove keys with "__"

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = torch.from_numpy(np.array(list(batched[key])))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
            else:
                result[key] = list(batched[key])
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = torch.from_numpy(np.stack(list(batched[key])))
        else:
            result[key] = list(batched[key])
    return result


def decode_npy(b: bytes):
    with io.BytesIO(b) as f:
        return np.load(f, allow_pickle=True)


def augment(clip, size=512):
    # Assumes that the videos are already square
    clip = TVT.functional.resize(clip, [size, size], interpolation=TVT.InterpolationMode.BICUBIC, antialias=True)
    clip = (clip - 0.5) / 0.5
    clip = clip.clamp(-1.0, 1.0)  # to prevent values outside from [-1,1] in bicubic mode
    return clip


class TrackerShardsDataModule:
    def __init__(
        self,
        tar_base: str,
        batch_size: int,
        num_workers: int = 8,
        shuffle: int = 1000,
        min_num_tracks: int = 128,
        context_length: int = 128,
        num_targets_per_context: int = 15,
        allow_poke_query_overlap: bool = True,
        skip: int = 25,
        allow_invisible_track_ends: bool = True,
        allow_out_of_frame_track_ends: bool = True,
        # Static camera detection heuristic
        static_camera_flow_mag_threshold: float = 0.01,
        static_camera_fraction_threshold: float = 0.4,
        # Other
        verbose: bool = False,
    ):
        super().__init__()
        self.tar_base = tar_base
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.min_num_tracks = min_num_tracks
        self.context_length = context_length
        self.num_targets_per_context = num_targets_per_context
        self.allow_poke_query_overlap = allow_poke_query_overlap
        self.skip = skip
        self.allow_invisible_track_ends = allow_invisible_track_ends
        self.allow_out_of_frame_track_ends = allow_out_of_frame_track_ends

        self.static_camera_flow_mag_threshold = static_camera_flow_mag_threshold
        self.static_camera_fraction_threshold = static_camera_fraction_threshold

        self.verbose = verbose

    def extract_training_sample(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | int | bool]:
        try:
            visibility: Bool[torch.Tensor, "t n_t"] = sample["visibility"]
            tracks: Float[torch.Tensor, "t n_t 2"] = sample["tracks"]
            track_in_frame: Bool[torch.Tensor, "t n_t"] = (
                (tracks[..., 0] >= 0) & (tracks[..., 0] <= 1) & (tracks[..., 1] >= 0) & (tracks[..., 1] <= 1)
            )
            visible_and_in_frame: Bool[torch.Tensor, "t n_t"] = visibility & track_in_frame
            valid_start_frames = (
                visible_and_in_frame.int().sum(dim=1) >= self.min_num_tracks
            )  # Ones that have at least `num_tracks` visible tracks
            if not valid_start_frames.any():
                return {"valid": False}
            # skips: Float[torch.Tensor, "t_start t_end"] = sample["times"][None, :] - sample["times"][:, None]
            # valid_skips: Bool[torch.Tensor, "t_start t_end"] = (
            #     (skips >= self.time_skip_min) & (skips <= self.time_skip_max)
            # ) & valid_start_frames[:, None]
            # if not valid_skips.any():
            #     return {"valid": False}

            # Select a random valid start & end frame pair
            # i_start, i_end = valid_skips.nonzero()[torch.randint(valid_skips.sum(), (1,))].squeeze()

            valid_start_frames = valid_start_frames & (
                torch.arange(valid_start_frames.size(0)) < valid_start_frames.size(0) - self.skip
            )
            if not valid_start_frames.any():
                if self.verbose:
                    print("No valid start frames found after skipping.")
                return {"valid": False}
            i_start = valid_start_frames.nonzero()[torch.randint(valid_start_frames.sum(), (1,))].squeeze()
            i_end = i_start + self.skip

            valid_tracks_mask = visible_and_in_frame[i_start]
            if not self.allow_invisible_track_ends:
                valid_tracks_mask &= visibility[i_end]
            if not self.allow_out_of_frame_track_ends:
                valid_tracks_mask &= track_in_frame[i_end]

            valid_tracks = valid_tracks_mask.nonzero().flatten()
            if len(valid_tracks) < self.min_num_tracks:
                if self.verbose:
                    print(f"Not enough valid tracks: {len(valid_tracks)} < {self.min_num_tracks}")
                return {"valid": False}
            # Get all valid tracks (randomly shuffled)
            idxs = valid_tracks[torch.randperm(len(valid_tracks))]
            pos: Float[torch.Tensor, "l 2"] = tracks[i_start, idxs]
            flow: Float[torch.Tensor, "l 2"] = tracks[i_end, idxs] - pos
            return {
                "i_frame": int(i_start.item()),
                "pos": pos,  # [l, 2] in [0, 1]
                "flow": flow,  # [l, 2] in ~[-1, 1]
            }
        except Exception as e:
            print(f"Error while extracting training data from sample: {e}")
            return {"valid": False}

    def get_camera_static(self, sample: dict[str, torch.Tensor]) -> Bool[torch.Tensor, ""]:
        flow: Float[torch.Tensor, "l 2"] = sample["flow"]
        return (
            flow.norm(dim=-1) < self.static_camera_flow_mag_threshold
        ).float().mean() > self.static_camera_fraction_threshold

    def build_targets(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | bool]:
        try:
            camera_static = self.get_camera_static(sample)

            # We generally take the first context_length indices as pokes
            # The queries are randomly chosen from the rest, either overlapping with the pokes or not
            pos: Float[torch.Tensor, "l 2"] = sample.pop("pos")
            flow: Float[torch.Tensor, "l 2"] = sample.pop("flow")
            L = pos.size(0)
            if L < self.context_length + (self.num_targets_per_context if self.allow_poke_query_overlap else 0):
                if self.verbose:
                    print(f"Not enough data points: {L} < {self.context_length + self.num_targets_per_context}")
                return {"valid": False}
            pos_query, flow_query = [], []
            for i in range(self.context_length):
                offset = random.randrange(
                    0 if self.allow_poke_query_overlap else i, L - i - self.num_targets_per_context
                )
                pos_query.append(pos[offset : offset + self.num_targets_per_context])
                flow_query.append(flow[offset : offset + self.num_targets_per_context])
            return sample | {
                "pos_poke": pos[: self.context_length],  # [n_p, 2] in [0, 1]
                "flow_poke": flow[: self.context_length],  # [n_p, 2] in ~[-1, 1]
                "pos_query": torch.stack(pos_query, dim=0),  # [n_p, n_q, 2] in [0, 1]
                "flow_query": torch.stack(flow_query, dim=0),  # [n_p, n_q, 2] in ~[-1, 1]
                "camera_static": camera_static,  # []
            }  # type: ignore

        except Exception as e:
            print(f"Error while building targets from sample: {e}")
            return {"valid": False}

    def make_loader(self):
        def construct_sample(sample: dict[str, bytes]) -> dict[str, torch.Tensor | bool]:
            try:
                required_keys = ["video.mpg", "tracks.npy", "visibility.npy"]
                assert all(k in sample for k in required_keys), f"Expected keys {required_keys}, got {sample.keys()}"

                d = {
                    "tracks": torch.from_numpy(decode_npy(sample["tracks.npy"])),  # [t, n_t, 2]
                    "visibility": torch.from_numpy(decode_npy(sample["visibility.npy"])),  # [t, n_t]
                }

                sample_out = self.extract_training_sample(d)
                if not sample_out.get("valid", True):
                    return {"valid": False}
                sample_out = self.build_targets(sample_out)
                if not sample_out.get("valid", True):
                    return {"valid": False}

                i_f = sample_out.pop("i_frame")
                # Retrieve target frame from video
                with io.BytesIO(sample["video.mpg"]) as buf, av.open(buf) as container:
                    c_f = 0
                    target_frame = None
                    for packet in container.demux():
                        if not target_frame is None:
                            break
                        for frame in packet.decode():
                            if c_f == i_f:
                                target_frame = frame.to_ndarray(format="rgb24")
                                break
                            c_f += 1
                assert c_f == i_f, f"{i_f=}, {c_f=}"
                # Resize and normalize the target frame
                # TODO: augment -> explicit
                x: Float[torch.Tensor, "c h w"] = augment(
                    einops.rearrange(torch.from_numpy(target_frame).float() / 255, "h w c -> 1 c h w"), size=512
                )[
                    0
                ]  # [-1, 1]
                return sample_out | {
                    "x": x,
                }
            except Exception as e:
                print(f"Error while constructing sample : {e}")
                return {"valid": False}

        dataset = wds.DataPipeline(
            wds.ResampledShards(urls=([str(f) for f in Path(self.tar_base).rglob("*.tar")])),
            wds.detshuffle(),
            wds.split_by_node,
            wds.split_by_worker,
            partial(wds.tarfile_samples, handler=wds.warn_and_continue),
            *([wds.shuffle(self.shuffle)] if self.shuffle != 0 else []),
            wds.map(construct_sample),
            wds.select(lambda s: s.get("valid", True)),
            wds.batched(self.batch_size, partial=False, collation_fn=dict_collation_fn),
        )
        return wds.WebLoader(dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True)
