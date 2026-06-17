<p align="center">
  <img src="assets/teaser.png" alt="MiniT2I logo" />
</p>

<h2 align="center">MiniT2I: A Minimalist Baseline for Text-to-Image Generation</h2>

<p align="center">
  <a href="https://peppaking8.github.io/#/post/text-to-image-generation-made-simple"><img src="https://img.shields.io/badge/Blog-MiniT2I-2ea44f.svg" alt="MiniT2I blog post" /></a>
  &nbsp;
  <a href="https://huggingface.co/MiniT2I"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20JAX%20Checkpoints-MiniT2I-yellow.svg" alt="Hugging Face JAX checkpoints" /></a>
  &nbsp;
  <!-- <a href="https://github.com/PeppaKing8/minit2i-jax"><img src="https://img.shields.io/badge/Code-JAX-blue.svg" alt="JAX code" /></a>
  &nbsp; -->
  <a href="https://github.com/Hope7Happiness/minit2i-torch"><img src="https://img.shields.io/badge/Code-PyTorch-blue.svg" alt="PyTorch code" /></a>
  &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License" /></a>
</p>

Official JAX/Flax training implementation of **MiniT2I**.

MiniT2I is a simple direct-RGB text-to-image generator that trains a pixel-space MM-JiT denoiser with flow matching, conditioned on frozen FLAN-T5-Large text tokens. The recipe is intentionally plain: avoiding image tokenizers, cascaded generation, RL stages, and any auxiliary losses. Data used in training MiniT2I is fully public and easy to implement. For more details, please refer to our [blog post](https://peppaking8.github.io/#/post/text-to-image-generation-made-simple).

This repository contains the original JAX/Flax diffusion training and evaluation code used for MiniT2I. 

- For the JAX Mean Flow distillation code used by the four-step MiniT2I-B/16-MF checkpoint, use the [`mean_flow_distill`](https://github.com/PeppaKing8/minit2i-jax/tree/mean_flow_distill) branch.
- For a PyTorch/Diffusers implementation with inference and LoRA adaptation, see [`Hope7Happiness/minit2i-torch`](https://github.com/Hope7Happiness/minit2i-torch).

## Model Zoo

| Model | Params | Patch | GenEval | DPG-Bench | Hugging Face |
| --- | ---: | ---: | ---: |  ---: | --- |
| MiniT2I-B/16 | 258M + 341M text encoder | 16 | 0.873 | 84.2 | [MiniT2I-B-16](https://huggingface.co/MiniT2I/MiniT2I-B-16-jax) |
| MiniT2I-L/16 | 912M + 341M text encoder | 16 | 0.883 | 85.9 | [MiniT2I-L-16](https://huggingface.co/MiniT2I/MiniT2I-L-16-jax) |

The repository also includes our default baseline B/32 for ablation and reproduction studies.

## Repository Layout

```text
.
|-- main.py                     # JAX distributed train/eval entry point
|-- train.py                    # training loop, checkpointing, sampling, online eval
|-- diffusion.py                # flow-matching objective and samplers
|-- configs/                    # defaults plus b32/b16/l16 YAML recipes
|-- settings.py                 # local paths, checkpoints, eval assets, logging
|-- models/                     # MM-JiT, T5 encoder, Flax/Torch-compatible layers
|-- utils/                      # input pipeline, pjit sharding, checkpoints, logging
|-- evaluators/                 # FID, GenEval, and DPG-Bench dispatch
|-- external/                   # JAX evaluator/model ports used by benchmarks
`-- scripts/                    # install/train/eval launch helpers
```

## Installation

**This codebase is TPU-oriented** that uses JAX distributed initialization. 

Create a Python environment, install a TPU-compatible JAX build, then install the remaining dependencies:

```bash
python -m pip install "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -r requirements.txt
```

The same commands are wrapped by:

```bash
bash scripts/install.sh
```

Authenticate only with services needed by your run:

```bash
hf auth login
wandb login
```

Project-level paths, checkpoint roots, benchmark asset roots, and W&B defaults live in [`settings.py`](settings.py). Experiment-dependent fields such as `load_from` belong in run YAML configs or command-line overrides.

## Dataset Preparation

Training uses WebDataset tar shards. Each sample must contain an image under `jpg` or `png` and a caption under `txt`. The input pipeline resizes and center-crops images, normalizes them to `[-1, 1]`, and tokenizes captions with the configured frozen text encoder.

See [`scripts/datasets/README.md`](scripts/datasets/README.md) for more details.

## Training

We provide training config files for B/32, B/16, and L/16, respectively:
- For **B/32**, start from [`configs/b32/pretrain.yml`](configs/b32/pretrain.yml);
- For **B/16**, start from [`configs/b16/pretrain.yml`](configs/b16/pretrain.yml);
- For **L/16**, start from [`configs/l16/pretrain.yml`](configs/l16/pretrain.yml).

Set `CC12M_ROOT` in `settings.py`, then keep the YAML focused on the training recipe:

```yaml
eval_only: False

dataset:
  use_cc12m: True
  use_blip3_ft60k: False
  use_dalle3: False
  use_sharegpt4o: False

eval:
  on_training: False
```

Launch:

```bash
bash scripts/train.sh pretrain \
  --config configs/load_config.py:pretrain_b16 \
  --workdir /path/to/runs/minit2i-pretrain
```

For the B/32 ablation recipe:

```bash
bash scripts/train.sh pretrain_b32 \
  --config configs/load_config.py:pretrain_b32 \
  --workdir /path/to/runs/minit2i-b32-pretrain
```

For fine-tuning, pass the pretrained checkpoint with `--load_from` or set `load_from` in the run YAML, then enable the 120K mix sources:

```yaml
eval_only: False
load_from: /path/to/pretrained/checkpoint_or_run_dir

dataset:
  use_cc12m: False
  use_blip3_ft60k: True
  use_dalle3: True
  use_sharegpt4o: True
```

Launch:

```bash
bash scripts/train.sh finetune \
  --config configs/load_config.py:finetune_b16 \
  --workdir /path/to/runs/minit2i-finetune \
  --load_from /path/to/pretrained/checkpoint_or_run_dir
```

Checkpoints are saved through Flax checkpointing directly under `workdir`. Local absolute paths and `gs://` paths (GCS bucket) are both supported.

## Evaluation

Evaluation is driven by the same entry point with `eval_only: True`. The checkpoint is passed with `--load_from` when using [`scripts/eval.sh`](scripts/eval.sh), or with the experiment-level `load_from` field in a YAML config. It is restored in [`train.py`](train.py).

`load_from` may point either to a Flax checkpoint directory (starting with `checkpoint_`), or to a _parent directory_ containing Flax checkpoints. If a parent directory is given, the _latest_ `checkpoint_*` under that directory is restored.

### Download Checkpoints

Install the Hugging Face CLI if needed:

```bash
python -m pip install -U "huggingface_hub[cli]"
```

Download a JAX checkpoint:

```bash
hf download MiniT2I/MiniT2I-B-16-jax \
  --local-dir /path/to/checkpoints/MiniT2I-B-16-jax

hf download MiniT2I/MiniT2I-L-16-jax \
  --local-dir /path/to/checkpoints/MiniT2I-L-16-jax
```

The model architecture in the YAML config must match the checkpoint for successful parameter loading:

| Checkpoint | Config |
| --- | --- |
| MiniT2I-B/16 ([`MiniT2I/MiniT2I-B-16-jax`](https://huggingface.co/MiniT2I/MiniT2I-B-16-jax)) | [`configs/load_config.py:eval_b16`](configs/load_config.py) |
| MiniT2I-L/16 ([`MiniT2I/MiniT2I-L-16-jax`](https://huggingface.co/MiniT2I/MiniT2I-L-16-jax)) | [`configs/load_config.py:eval_l16`](configs/load_config.py) |

### Run Evaluation

For B/16:

```bash
bash scripts/eval.sh eval \
  --config configs/load_config.py:eval_b16 \
  --workdir /path/to/runs/minit2i-eval \
  --load_from /path/to/checkpoints/MiniT2I-B-16-jax
```

For L/16:

```bash
bash scripts/eval.sh eval_l \
  --config configs/load_config.py:eval_l16 \
  --workdir /path/to/runs/minit2i-l-eval \
  --load_from /path/to/checkpoints/MiniT2I-L-16-jax
```

The combined evaluator currently wires:

- FID on MSCOCO-30K when enabled.
- GenEval with the JAX Mask2Former detector and optional CLIP color classifier (This JAX version is our reproduction).
- DPG-Bench with the JAX mPLUG VQA evaluator (This JAX version is our reproduction).

### Evaluation Assets

We evaluate MSCOCO-FID-30K, [GenEval](https://github.com/djghosh13/geneval), and [DPGBench](https://github.com/TencentQQGYLab/ELLA). For each benchmark, set the asset paths in `settings.py`, then enable the benchmark via setting `enable: True` inside the config yaml file. You can specify the CFG scale for each benchmark. For instance, if using GenEval with guidance scale `6.0`:

```yaml
eval:
  geneval:
    enable: True
    cfg_scale: 6.0
```
If a benchmark-specific value is missing, the evaluator falls back to the top-level `eval.cfg_scale`.

## Sampling

During training or eval-only runs, setting `eval_show_sample: True` writes samples from the built-in visualization prompts. If W&B or TensorBoard logging is disabled, images are saved under:

```text
<workdir>/writed_images/
```

## Acknowledgments

This codebase builds on a number of open-source efforts:

- Our GenEval and DPGBench evaluations on JAX are the re-implementation of the original evaluations from [`djghosh13/geneval`](https://github.com/djghosh13/geneval) and [`Jialuo21/DPG-Bench`](https://huggingface.co/datasets/Jialuo21/DPG-Bench).
- Public training data from [`CaptionEmporium/conceptual-captions-cc12m-llavanext`](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext), [`BLIP3o/BLIP3o-60k`](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k), [`CaptionEmporium/dalle3-llama3.2-11b`](https://huggingface.co/datasets/CaptionEmporium/dalle3-llama3.2-11b), [`FreedomIntelligence/ShareGPT-4o-Image`](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image), and [`lambdalabs/naruto-blip-captions`](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions).

We also thank GCP for providing the computational resources.

## Citation

```bibtex
@misc{minit2i2026,
  title  = {MiniT2I: A Minimalist Baseline for Text-to-Image Generation},
  author = {Wang, Xianbang and Zhao, Hanhong and Lu, Yiyang and Zhou, Kangyang and Ma, Linrui and He, Kaiming},
  year   = {2026},
  url    = {https://peppaking8.github.io/#/post/minit2i}
}
```
