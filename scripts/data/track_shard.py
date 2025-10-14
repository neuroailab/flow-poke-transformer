# SPDX-License-Identifier: MIT

import av
import click
import einops
import ffmpeg  # `https://github.com/kkroening/ffmpeg-python`, not `https://github.com/jonghwanhyeon/python-ffmpeg`!
import gc
import io
import numpy as np
import os
import random
import torch
import torchvision.transforms.v2 as TVT
import webdataset as wds

from functools import partial
from pathlib import Path
from tqdm.auto import tqdm

# set these two decord lines before `import torch` and weird stuff can happen. including `Segmentation fault (core dumped)`
import decord
decord.bridge.set_bridge("torch")


MP4_SUFFIX = "mp4"
MPEG_TRANSCODING_FRAMERATE = 60  # don't change this


def init_cotracker(
    repo_url: str = "facebookresearch/co-tracker",
    model_name: str = "cotracker3_offline",
    device: torch.device = torch.device("cuda"),
) -> torch.nn.Module:
    cotracker = torch.hub.load(repo_url, model_name)
    cotracker = cotracker.to(device=torch.device("cpu"))
    cotracker = cotracker.to(device=device)
    return cotracker


def select_frames(
    num_frames,
    sequence_length,
    frame_skip=0,
):
    indices = np.arange(num_frames)
    gather_indices = np.arange(sequence_length) * (frame_skip + 1)
    inds = indices[gather_indices]
    return inds


def augment(
    clip,
    interpolation=TVT.InterpolationMode.BICUBIC,
    size=512,
    center_crop=True,
) -> torch.Tensor:
    if center_crop:
        H, W = clip.shape[-2], clip.shape[-1]
        L = min(H, W)
        starth = H // 2 - (L // 2)
        startw = W // 2 - (L // 2)
        clip = clip[..., starth : (starth + L), startw : (startw + L)]
    clip = TVT.functional.resize(clip, (size, size), interpolation=interpolation, antialias=True)
    # normalize
    clip = (clip - 0.5) / 0.5 # [0, 1] -> [-1, 1]
    clip = clip.clamp(-1.0, 1.0)  # to prevent values outside from [-1,1] in bicubic mode
    return einops.rearrange(clip, "t c h w -> c t h w")


def load_mp4(
    data,
    min_sequence_length=0,
    select_args={},
    augment_args={},
    frame_skip=0,
    max_sequence_length=None,
) -> tuple[dict[str, any], int]:
    data_dict = {}
    status = -1
    with io.BytesIO(data) as data_io:
        vr = None
        try:
            vr = decord.VideoReader(data_io)
            num_frames = len(vr)
            if min_sequence_length is None:
                min_sequence_length = num_frames
            required_sequence_length = min_sequence_length * (frame_skip + 1) - frame_skip
            if num_frames < required_sequence_length:
                raise ValueError(f"Too few frames {num_frames=} < {required_sequence_length=}")
            sequence_length = num_frames // (frame_skip + 1)
            if max_sequence_length is not None and sequence_length > max_sequence_length:
                sequence_length = max_sequence_length
            inds = select_frames(
                num_frames,
                sequence_length,
                frame_skip=frame_skip,
                **select_args,
            )
            sequence = vr.get_batch(inds)
            sequence = einops.rearrange(sequence, "t h w c -> t c h w")
            # [0,255] -> [0,1]
            sequence = sequence.float() / 255.0
            if len(sequence) < len(inds):
                status = 0
                raise ValueError(f"Mismatch {len(sequence)} {len(inds)}")
            sequence = augment(sequence, **augment_args)
            data_dict["x"] = sequence
            status = 1
        except Exception as e:
            print(f"Caught {e}.", flush=True)
    if vr is not None:
        del vr
    gc.collect()
    return data_dict, status


def decode_mp4(
    sample:dict[str, any],
    **kwargs,
) -> dict[str, any]:
    return_dict = {}
    found_valid_ext = False
    assert "__key__" in sample.keys(), f"{sample.keys()}"
    key = sample["__key__"]
    for k, v in sample.items():
        if k.endswith(MP4_SUFFIX):
            found_valid_ext = True
            data, status = load_mp4(v, **kwargs)
            if status <= 0:
                return None
            else:
                for k2, v2 in data.items():
                    return_dict[k2] = v2
        else:
            return_dict[k] = v
    if not found_valid_ext:
        print(f"couldn't find valid video extension ({MP4_SUFFIX}) in {key=}")
        return None
    return return_dict


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True, **kwargs) -> dict[str, any]:
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}  # remove keys with "__"
    for s in samples:
        [batched[key].append(s[key]) for key in batched]
    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = torch.tensor(np.array(list(batched[key])))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
            else:
                result[key] = list(batched[key])
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = torch.tensor(np.stack(list(batched[key])))
        else:
            result[key] = list(batched[key])
    return result


def run_cotracker(
    video: torch.Tensor,  # B C T H W shape, [-1,1] values
    cotracker: torch.nn.Module,
    device: torch.device,
    t_chunk: int, # number of frames / size of video chunk along temporal axis
    nr_queries: int, # number of CoTracker queries
    frame_size: int, # size of frames `H == W`
) -> tuple[torch.Tensor, torch.Tensor]:
    input_video = einops.rearrange(video, "b c t h w -> b t c h w")
    input_video = input_video.to(device)
    B = input_video.shape[0]  # ideally equal to `batch_size` all the time. but the final batch can be smaller, so reading `B` here is safer.
            # format `(B N) 3`
    queries: torch.Tensor = torch.stack(
        [
            torch.randint(
                0, t_chunk, (nr_queries * B,),
                dtype=torch.float32, device="cuda",
            ),
            torch.randint(
                0, frame_size, (nr_queries * B,),
                dtype=torch.float32, device="cuda",
            ),
            torch.randint(
                0, frame_size, (nr_queries * B,),
                dtype=torch.float32, device="cuda",
            ),
        ],
        dim=-1,
    )
    queries = einops.rearrange(queries, "(B N) C -> B N C", B=B, N=nr_queries, C=3)
    with torch.no_grad():
        tracks, visibility = cotracker(
            input_video,
            queries=queries,
            backward_tracking=True,
        )
        tracks = tracks / frame_size
        tracks = tracks.float().cpu().numpy()
        visibility = visibility.to(dtype=torch.bool).cpu().numpy()
    return tracks, visibility


def transcode(
    frames,
    input_args: list = ["pipe:"],
    input_kwargs: dict = {
        "format": "rawvideo",
        "pix_fmt": "rgb24",
        "framerate": MPEG_TRANSCODING_FRAMERATE,
    },
    output_args: list = ["pipe:"],
    output_kwargs: dict = {
        "format": "mpeg",
        "vcodec": "mpeg1video",
        "qscale": 2,  # Set the quality scale (lower is better quality)
        "video_bitrate": "5000k",
    },
) -> bytes:
    t, h, w, c = frames.shape
    assert c == 3, f"{c=}"
    process = ffmpeg.input(
        *input_args,
        **input_kwargs,
        s="{}x{}".format(w, h),
    )
    process = process.output(
        *output_args,
        **output_kwargs,
    )
    process = process.overwrite_output()
    process = process.run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    stdout_data, stderr_data = process.communicate(input=frames.tobytes())
    stderr = stderr_data.decode("utf-8")
    if "Error" in stderr:
        print(f"Got error while re-encoding: {stderr}", flush=True)
        print("Skipping sample", flush=True)
        return None    
    with io.BytesIO() as video_buffer:
        video_buffer.write(stdout_data)
        process.wait()
        video_buffer.seek(0)
        buffer = video_buffer.getvalue()
    nr_written_frames = 0
    with io.BytesIO(buffer) as buf, av.open(buf) as container:
        for packet in container.demux():
            for frame in packet.decode():
                nr_written_frames = nr_written_frames + 1
    assert nr_written_frames == t, f"Number of frames has changed from {t=} to {nr_written_frames}"
    return buffer


def wds_filter(sample: dict | None) -> bool:
    # in case of failure `sample is None` -> returns `False` and is removed.
    return sample is not None


@click.command()
@click.argument("input_filename", type=click.Path(exists=True, path_type=Path))
@click.argument("output_filename", type=click.Path(exists=False, path_type=Path))
@click.option("--device", type=str, default="cuda:0")
@click.option("--frame_size", type=int, default=512)
@click.option("--frame_skip", type=int, default=1)
@click.option("--min_seq_len", type=int, default=12)
@click.option("--max_seq_len", type=int, default=24)
@click.option("--batch_size", type=int, default=1)
@click.option("--nr_workers", type=int, default=1)
@click.option("--no_tqdm", is_flag=True)
@click.option("--refresh_rate", type=int, default=100)
@click.option("--nr_queries", type=int, default=512)
@click.option("--video_key", type=str, default="video.mpg")
def track_shard(
    input_filename: Path,
    output_filename: Path,
    device: str = "cuda:0",
    init_cotracker_kwargs: dict = {},
    frame_size: int = 512,
    frame_skip: int = 1,
    min_seq_len: int = 12,
    max_seq_len: int = 24,
    batch_size: int = 1,
    nr_workers: int = 1,
    no_tqdm: bool = False,
    refresh_rate: int = 100,
    nr_queries: int = 512,  # CoTracker queries
    video_key: str = "video.mpg",
):
    device = torch.device(device)
    cotracker = init_cotracker(**init_cotracker_kwargs, device=device)
    assert os.path.isfile(str(input_filename)), f"{input_filename=}"
    dataset = wds.WebDataset(str(input_filename))
    dataset = dataset.decode()
    dataset = dataset.map(
        partial(
            decode_mp4,
            frame_skip=frame_skip,
            min_sequence_length=min_seq_len,
            augment_args={
                "size": frame_size,
                "center_crop": True,
            },
            max_sequence_length=max_seq_len * 2 if not max_seq_len is None else None, # Simple heuristic
        )
    )
    dataset = dataset.select(wds_filter)
    dataset = dataset.batched(batch_size, collation_fn=dict_collation_fn)
    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=nr_workers,
    )
    os.makedirs(str(output_filename.parent), exist_ok=True)
    with wds.TarWriter(str(output_filename)) as sink:
        for sample in tqdm(
            loader,
            desc=f"Preprocessing {input_filename}",
            disable=no_tqdm,
            miniters=refresh_rate,
        ):
            sample: dict[str, any]
            # format `b c t h w`
            video: torch.Tensor = sample["x"].float()
            T = video.shape[2]
            offsets = [
                offset
                for offset in range(0, T, max_seq_len)
                if min(max_seq_len, T - offset) >= min_seq_len
            ]
            if len(offsets) == 0:
                print(f"No suitable chunk found, got {T=} frames and need chunks in range {min_seq_len=} {max_seq_len=}", flush=True)
                continue
            t_start = random.choice(offsets)
            t_chunk = min(max_seq_len, T - t_start)
            video = video[:, :, t_start : t_start + t_chunk]
            tracks, visibility = run_cotracker(
                video=video,
                cotracker=cotracker,
                device=device,
                t_chunk=t_chunk,
                nr_queries=nr_queries,
                frame_size=frame_size,
            )
            video = einops.rearrange(video, "b c t h w -> b t h w c")
            video = video.cpu().numpy()
            video = (video + 1) * 127.5 # [-1,1] to [0,255] range
            video = video.round().astype(np.uint8)
            for b in range(video.shape[0]):
                frames = video[b]
                frames = np.ascontiguousarray(frames) # for sanity...
                try:
                    buffer = transcode(frames)
                    if buffer is None:
                        continue
                    data = {
                        "tracks.npy": tracks[b],
                        "visibility.npy": visibility[b],
                        video_key: buffer,
                    }
                    for k, v in sample.items():
                        val = v[b]
                        if k not in ["x"]:
                            if k.startswith("__"):
                                data[k] = val
                    sink.write(data)
                except Exception as e:
                    # print(f"Caught {e=}", flush=True)
                    # continue
                    raise
                del frames
            del video, sample


if __name__ == "__main__":
    track_shard()
