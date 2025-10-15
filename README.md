# Flow Poke Transformer
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://compvis.github.io/flow-poke-transformer/)
[![Paper](https://img.shields.io/badge/arXiv-paper-b31b1b)](https://arxiv.org/abs/2510.12777)
[![Weights](https://img.shields.io/badge/HuggingFace-Weights-orange)](https://huggingface.co/CompVis/flow-poke-transformer)
<h2 align="center"><i>What If:</i> Understanding Motion Through Sparse Interactions</h2>
<div align="center"> 
  <a href="https://stefan-baumann.eu/" target="_blank">Stefan A. Baumann</a><sup>*</sup> · 
  <a href="https://nickstracke.dev/" target="_blank">Nick Stracke</a><sup>*</sup> · 
  <a href="" target="_blank">Timy Phan</a><sup>*</sup> · 
  <a href="https://ommer-lab.com/people/ommer/" target="_blank">Björn Ommer</a>
</div>
<p align="center"> 
  <b>CompVis @ LMU Munich, MCML</b><br/>ICCV 2025
</p>

![FPT predicts distributions of potential motion for sparse points](docs/static/images/teaser_fig.png)

Flow Poke Transformer (FPT) directly models the uncertainty of the world by predicting distributions of how objects (<span style="color:#ff7f0e">×</span>) may move conditioned on some input movements (pokes, →). We see that whether the hand (below paw) or the paw (above hand) moves downwards directly influences the other's movement. Left: the paw pushing the hand down, will force the hand downwards, resulting in a unimodal distribution. Right: the hand moving down results in two modes, the paw following along or staying put.

This codebase is a minimal PyTorch implementation covering training & various inference settings.

## 🚀 Usage
The easiest way to try FPT is via our interactive demo, which you can launch as:
```shell
python -m scripts.demo.app --compile True --warmup_compiled_paths True
```
Compilation is optional, but recommended for a better time using the UI. A checkpoint will be downloaded from huggingface by default if not explicitly specified via the CLI.

When using it yourself, the simplest way to use it is via `torch.hub`:
```python
model = torch.hub.load("CompVis/flow_poke_transformer", "fpt_base")
```

If you want to completely integrate FPT into your own codebase, copy `model.py` and `dinov2.py` to your codebase and you should effectively be good to go. Then instantiate the model as
```python
model: FlowPokeTransformer = FlowPokeTransformer_Base()
state_dict = torch.load("fpt_base.pt")
model.load_state_dict(state_dict)
model.requires_grad_(False)
model.eval()
```

The `FlowPokeTransformer` class contains all the methods that you should need to use FPT in various applications. For high-level usage, use the `FlowPokeTransformer.predict_*()` methods. For low-level usage, the module's `forward()` can be used.

The only dependencies you should need are a recent `torch` (to enable flex attention, although it would be plausible to patch it out with some effort to enable usage of lower torch version), and any `einops`, `tqdm`, and `jaxtyping` (dependency can be removed by deleting type hints) versions.

### About the Codebase
Code files are separated into major blocks with extensive comments explaining relevant choices, details, and conventions.
For all public-facing APIs involving tensors, type hints with [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping) are provided, which might look like this: `img: Float[torch.Tensor, "b c h w"]`. They annotate the dtype (`Float`), tensor type `torch.Tensor`, and shape `b c h w`, and should (hopefully) make the code fully self-explanatory.

**Coordinate & Image Conventions.**
We represent coordinates in (x, y) order with image coordinates normalized in $[0, 1]^2$ (the outer bounds of the image are defined to be 0 and 1 and coordinates are assigned based on pixel centers).
Flow is in the same coordinate system, resulting in $[-1, 1]^2$ flow.
Pixel values are normalized to $[-1, 1]$.
See the `Attention & RoPE Utilities` section in [`model.py`](flow_poke/model.py) for further details


## 🔧 Training

**Data Preprocessing.**
For data preprocessing instructions, please refer to the [corresponding readme](scripts/data/README.md).

**Launching Training.**
Single-GPU training can be launched via
```shell
python train.py --data_tar_base /path/to/preprocessed/shards --out_dir output/test --compile True
```
Similarly, multi-GPU training, e.g., on 2 GPUs, can be launched using torchrun:
```shell
torchrun --nnodes 1 --nproc-per-node 2 train.py [...]
```
Training can be continued from a previous checkpoint by specifying, e.g., `--load_checkpoint output/test/checkpoints/checkpoint_0100000.pt`.
Remove `--compile True` for significantly faster startup time at the cost of slower training & significantly increased VRAM usage.

For a full list of available arguments, refer to [`train.train()`](train.py) method. We use [`fire`](https://github.com/google/python-fire), such that every argument to the main train function is directly available as a CLI argument.

## Models
We release the weights of our open-set model via huggingface at https://huggingface.co/CompVis (under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) license), and will potentially release further variants (scaled up or with other improvements).
Due to [copyright concerns surrounding the WebVid dataset](https://github.com/m-bain/webvid?tab=readme-ov-file#dataset-no-longer-available-but-you-can-still-use-it-for-internal-non-commerical-purposes), will not distribute the model weights for the model trained on it. Both models perform approximately equally (see Tab. 1 in the paper), although this will vary on a case-by-case basis due to the different training data.

## Code Credit
- Some model code is adapted from [k-diffusion](https://github.com/crowsonkb/k-diffusion) by Katherine Crowson (MIT)
- The DINOv2 code is adapted from [minDinoV2](https://github.com/cloneofsimo/minDinoV2) by Simo Ryu, which is in turn adapted from the [official implementation](https://github.com/facebookresearch/dinov2/) by Oquab et al. (Apache 2.0)

## 🎓 Citation
If you find our model or code useful, please cite our paper:
```bibtex
@inproceedings{baumann2025whatif,
    title={What If: Understanding Motion Through Sparse Interactions}, 
    author={Stefan Andreas Baumann and Nick Stracke and Timy Phan and Bj{\"o}rn Ommer},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025}
}
```
