# SPDX-License-Identifier: MIT

from pathlib import Path
import random

import click
import webdataset as wds
from tqdm.auto import tqdm


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(exists=False, path_type=Path))
@click.option("--max-size", type=int, default=1e9)
@click.option("--prefix", type=str, default="shard")
def shard(input_path: Path, output_path: Path, max_size: int, prefix: str = "shard"):
    output_path = Path(output_path.decode("utf-8"))
    input_path = Path(input_path.decode("utf-8"))
    files = list(input_path.glob("**/*.mp4"))
    idxs = list(range(len(files)))
    random.Random(42).shuffle(idxs)
    assert not output_path.exists() or not any(f.name.startswith(prefix) for f in output_path.iterdir())
    output_path.mkdir(parents=True, exist_ok=True)
    with wds.ShardWriter(f"{output_path}/{prefix}_%06d.tar", maxsize=max_size) as sink:
        for i in tqdm(idxs):
            path = files[i]
            with open(path, "rb") as f:
                data = f.read()
            sink.write(
                {
                    "__key__": f"{str(path.relative_to(input_path)).replace('/', '_').replace('.mp4', '')}",
                    "mp4": data,
                }
            )


if __name__ == "__main__":
    shard()
