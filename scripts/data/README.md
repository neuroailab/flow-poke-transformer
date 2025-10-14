# Flow Poke Transformer Data Preprocessing
The flow poke transformer training relies on large-scale video datasets with pre-extracted trackers.
Both the dataset and tracking method are generally exchangeable, no aspect in the model depends on specific choices.

We generally use sharded data where multiple samples are combined into "shards" using [`webdataset`](https://github.com/webdataset/webdataset) to reduce load on file servers in HPC systems.

## Preparing a Dataset Yourself
0. Collect a set of videos for your target domain.
1. Install additional requirements for running the preprocessing listed in `requirements.txt`.
2. Create shards from the video datasets using `python shard_videos.py /path/to/videos /path/to/output/shards`. This script assumes that videos are stored as `mp4` files. The glob pattern in the code can be adapted to change this. Just make sure that the tracking script in the next step can decode your videos.
3. For each shard, perform tracking. We provide a reference script using CoTracker3 that can be run by invoking `python track_shard.py /path/to/shard.tar /path/to/preprocessed/output/shard.tar`. This will use the first visible CUDA GPU to perform the preprocessing by default.
