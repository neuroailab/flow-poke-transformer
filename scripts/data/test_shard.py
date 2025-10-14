import av
import click
import colorsys
import einops
import io
import numpy as np
import os
import random
import torch
import webdataset as wds

from pathlib import Path
from tqdm.auto import tqdm


MPEG_SUFFIX = "mpg"
EXPECTED_KEYS = {
    "vis": "visibility.npy",
    "track": "tracks.npy",
    "vid": "video.mpg",
}


def load_mpeg(
    data: bytes,
    key: str,
):
    try:
        frames = []
        with io.BytesIO(data) as buf, av.open(buf) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))

        frames = np.stack(frames)
        frames = torch.from_numpy(frames)
        frames = einops.rearrange(frames, "t h w c -> c t h w")
        frames = (frames / 127.5 - 1.)
        frames = frames.clamp(-1,1)
    except Exception as exc:
        print(f"Caught exception {exc=} at {key=}")
        return None
    return frames


def decode_mpeg(
    sample:dict[str, any],
    **kwargs,
):
    return_dict = {}
    found_valid_ext = False
    assert "__key__" in sample.keys(), f"{sample.keys()}"
    key = sample["__key__"]
    for k, v in sample.items():
        if k.endswith(MPEG_SUFFIX):
            found_valid_ext = True
            data = load_mpeg(v, key=key)
            if data is None:
                return None
            else:
                return_dict[k] = data
        else:
            return_dict[k] = v
    if not found_valid_ext:
        print(f"couldn't find valid video extension ({MPEG_SUFFIX}) in {key=}")
        return None
    return return_dict


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True, **kwargs):
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


def wds_filter(sample: dict | None):
    # in case of failure sample is None
    return sample is not None


def get_color(
    hue: float,  # [0,1]
    t_rel: int,  # [0,N] N is `norm_scale`
    norm_scale: int,
    min_intensity: float, # [0, 1]
):
    intensity = (t_rel + 1) * norm_scale  # [0,1] range
    realigned_intensity = min_intensity + (1 - min_intensity) * intensity # [0,1] range
    rgb = colorsys.hsv_to_rgb(hue, realigned_intensity, realigned_intensity)
    return rgb


def get_colors(num_colors: int) -> list[tuple[int, int, int]]:
    # copied from: `https://github.com/facebookresearch/co-tracker/blob/main/gradio_demo/app.py#L23`
    """Gets colormap for points."""
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0
        color = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(
            (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        )
    random.shuffle(colors)
    return colors


def paint_point_track(
    frames: np.ndarray,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
    colormap:list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    # copied from: `https://github.com/facebookresearch/co-tracker/blob/main/gradio_demo/app.py#L95`
    """Converts a sequence of points to color code video.
    Args:
      frames: [num_frames, height, width, 3], np.uint8, [0, 255]
      point_tracks: [num_points, num_frames, 2], np.float32, [0, width / height]
      visibles: [num_points, num_frames], bool
      colormap: colormap for points, each point has a different RGB color.
    Returns:
      video: [num_frames, height, width, 3], np.uint8, [0, 255]
    """
    num_points, num_frames = point_tracks.shape[0:2]
    if colormap is None:
      colormap = get_colors(num_colors=num_points)
    height, width = frames.shape[1:3]
    # dot_size_as_fraction_of_min_edge = 0.015
    dot_size_as_fraction_of_min_edge = 0.005
    radius = int(round(min(height, width) * dot_size_as_fraction_of_min_edge))
    diam = radius * 2 + 1
    quadratic_y = np.square(np.arange(diam)[:, np.newaxis] - radius - 1)
    quadratic_x = np.square(np.arange(diam)[np.newaxis, :] - radius - 1)
    icon = (quadratic_y + quadratic_x) - (radius**2) / 2.0
    sharpness = 0.15
    icon = np.clip(icon / (radius * 2 * sharpness), 0, 1)
    icon = 1 - icon[:, :, np.newaxis]
    icon1 = np.pad(icon, [(0, 1), (0, 1), (0, 0)])
    icon2 = np.pad(icon, [(1, 0), (0, 1), (0, 0)])
    icon3 = np.pad(icon, [(0, 1), (1, 0), (0, 0)])
    icon4 = np.pad(icon, [(1, 0), (1, 0), (0, 0)])
    video = frames.copy()
    for t in range(num_frames):
        # Pad so that points that extend outside the image frame don't crash us
        image = np.pad(
            video[t],
            [
                (radius + 1, radius + 1),
                (radius + 1, radius + 1),
                (0, 0),
            ],
        )
        for i in range(num_points):
            # The icon is centered at the center of a pixel, but the input coordinates
            # are raster coordinates.  Therefore, to render a point at (1,1) (which
            # lies on the corner between four pixels), we need 1/4 of the icon placed
            # centered on the 0'th row, 0'th column, etc.  We need to subtract
            # 0.5 to make the fractional position come out right.
            x, y = point_tracks[i, t, :] + 0.5
            x = min(max(x, 0.0), width)
            y = min(max(y, 0.0), height)
            rgb = colormap[i]
            if visibles[i, t]:
                x1, y1 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
                x2, y2 = x1 + 1, y1 + 1
                # bilinear interpolation
                patch = (
                    icon1 * (x2 - x) * (y2 - y)
                    + icon2 * (x2 - x) * (y - y1)
                    + icon3 * (x - x1) * (y2 - y)
                    + icon4 * (x - x1) * (y - y1)
                )
                x_ub = x1 + 2 * radius + 2
                y_ub = y1 + 2 * radius + 2
                image[y1:y_ub, x1:x_ub, :] = (1 - patch) * image[
                    y1:y_ub, x1:x_ub, :
                ] + patch * np.array(rgb)[np.newaxis, np.newaxis, :]
        # Remove the pad
        video[t] = image[
            radius + 1 : -radius - 1, radius + 1 : -radius - 1
        ].astype(np.uint8)
    video = np.concatenate(
        [frames,video], axis=-2,
    )
    return video


def write_mp4(
    video: np.ndarray,
    path:str,
    fps: int = 24,
    vcodec: str = "libx264",
    pix_fmt: str = "yuv420p",
    print_success: bool = True,
):
    T, H, W, C = video.shape
    with av.open(path, mode="w") as container:
        stream = container.add_stream(vcodec, rate=fps)
        stream.width = W
        stream.height = H
        stream.pix_fmt = pix_fmt
        for t in range(T):
            frame = av.VideoFrame.from_ndarray(video[t], format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    if print_success:
        print(f"\nWritten to {path=}", flush=True)


@click.command()
@click.argument("input_filename", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dirname", type=click.Path(exists=False, path_type=Path))
@click.option("--batch_size", type=int, default=1)
@click.option("--nr_workers", type=int, default=1)
@click.option("--shuffle", type=int, default=0)
@click.option("--max_count", type=int, default=-1)
@click.option("--no_tqdm", is_flag=True)
@click.option("--no_denorm", is_flag=True)
def test_shard(
    input_filename: Path,
    output_dirname: Path,
    batch_size: int = 1,
    nr_workers: int = 1,
    shuffle: int = 0,
    max_count: int = -1,  # negative means ignored, non-negative means early stop when this thresh is reached
    no_tqdm: bool = False,
    no_denorm: bool = False,
):
    dataset = wds.WebDataset(str(input_filename))
    dataset = dataset.decode()
    dataset = dataset.map(decode_mpeg)
    dataset = dataset.select(wds_filter)
    dataset = dataset.shuffle(shuffle)
    dataset = dataset.batched(batch_size, collation_fn=dict_collation_fn)
    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=nr_workers,
    )
    os.makedirs(str(output_dirname), exist_ok=True)
    count = 0
    for sample in tqdm(
        loader,
        desc=f"Visualizing {input_filename}",
        disable=no_tqdm,
    ):
        if max_count >= 0 and count >= max_count:
            break
        count = count + 1
        sample: dict[str, any]
        for k,v in EXPECTED_KEYS.items():
            assert v in sample.keys(), f"failed to find key {v} in {sample.keys()}"
        key = sample["__key__"]
        # shape `B C T H W`, dtype=float32, vals `[-1, 1]` (strict)
        video = sample[EXPECTED_KEYS["vid"]].cpu().numpy()
        # shape `B T N 2` (`N` are the number of CoTracker queries), dtype=float32, vals `[0, 1]` (soft, 0-1 are borders)
        tracks = sample[EXPECTED_KEYS["track"]].cpu().numpy()
        # shape `B T N` (`N` are the number of CoTracker queries), dtype=bool, vals `{True, False}` (strict)
        visibility = sample[EXPECTED_KEYS["vis"]].cpu().numpy()
        # shape consistency checks. assumes `B` is always consistent, but sanity checks can always be added ofc.
        T, H, W = video.shape[-3:]
        N = visibility.shape[-1]
        assert H == W, f"{video.shape}"  # assuming square shape here
        assert T == tracks.shape[1], f"{T=} {tracks.shape}"
        assert N == tracks.shape[2], f"{N=} {tracks.shape}"
        assert 2 == tracks.shape[-1], f"{N=} {tracks.shape}"
        assert T == visibility.shape[1], f"{T=} {visibility.shape}"
        # render
        B = video.shape[0]
        video = (video + 1) * 127.5  # [-1,1] -> [0,255]
        video = video.round().astype(np.uint8)
        video = einops.rearrange(video, "B C T H W -> B T H W C")
        # normed [0,1] -> [0,H] (px units)
        # assumes `H==W`, otherwise would need to scale channels by `H` or `W` independently
        if not no_denorm:
            tracks = tracks * H
        tracks = einops.rearrange(tracks, "B T N C -> B N T C")
        visibility = einops.rearrange(visibility, "B T N -> B N T")
        for b in range(B):
            rendered = paint_point_track(
                frames=video[b],
                point_tracks=tracks[b],
                visibles=visibility[b],
            )
            write_mp4(video=rendered, path=os.path.join(str(output_dirname), f"{key[b]}.mp4"))


if __name__ == "__main__":
    test_shard()