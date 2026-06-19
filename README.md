<p align="center">
  <img src="assets/teaser.png" alt="MiniT2I logo" />
</p>

<h2 align="center">MiniT2I: A Minimalist Baseline for Text-to-Image Generation</h2>

<p align="center">
  <a href="https://peppaking8.github.io/#/post/minit2i"><img src="https://img.shields.io/badge/Blog-MiniT2I-2ea44f.svg" alt="MiniT2I blog post" /></a>
  &nbsp;
  <a href="https://huggingface.co/MiniT2I/MiniT2I-B-16-MF-jax"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-B%2F16--MF-yellow.svg" alt="Hugging Face Mean Flow checkpoint" /></a>
  &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License" /></a>
</p>

This branch contains the JAX/Flax **Mean Flow distillation** code used for the released four-step **MiniT2I-B/16-MF** checkpoint.

It is intentionally not a diffusion pretraining or fine-tuning release branch. For the original MiniT2I diffusion training and benchmark code, use the [`main`](https://github.com/PeppaKing8/minit2i-jax/tree/main) branch. For PyTorch/Diffusers inference and LoRA support, use [`Hope7Happiness/t2i-release`](https://github.com/Hope7Happiness/t2i-release).

## What This Branch Includes

```text
.
|-- main.py                         # JAX distributed MF train/eval entry point
|-- train_mean_flow.py              # checkpointing, resume, sampling, online eval
|-- configs/
|   |-- distill_config_b16.yml      # B/16 teacher-guided MF distillation
|   |-- eval_distill_config_b16.yml # B/16-MF sampling/benchmark eval
|   |-- load_config.py              # explicit alias/file config loader
|   `-- default.py                  # shared defaults
|-- models/
|   |-- mean_flow.py                # MF objective and four-step sampler
|   |-- mean_flow_mmjit.py          # MF student MM-JiT
|   `-- mmjit.py                    # frozen teacher MM-JiT architecture
|-- utils/                          # checkpoints, input pipeline, sharding, logging
|-- evaluators/                     # GenEval / DPG-Bench / optional FID wrappers
`-- external/                       # benchmark model ports
```

The branch keeps the MM-JiT teacher code because the distillation objective calls the frozen teacher during training. It removes the original diffusion training loop, diffusion sampler, and diffusion pretrain/fine-tune configs.

## Installation

This codebase is TPU-oriented and uses JAX distributed initialization.

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

Project-level dataset roots, checkpoint roots, benchmark asset paths, and W&B defaults live in [`settings.py`](settings.py). Experiment-dependent fields such as `load_from` and `load_pt_from` should be passed on the command line or in a run YAML.

## Checkpoints

Download the released B/16 diffusion teacher before starting a fresh distillation run:

```bash
hf download MiniT2I/MiniT2I-B-16-jax \
  --local-dir /path/to/checkpoints/MiniT2I-B-16-jax
```

Download the released four-step Mean Flow checkpoint for evaluation:

```bash
hf download MiniT2I/MiniT2I-B-16-MF-jax \
  --local-dir /path/to/checkpoints/MiniT2I-B-16-MF-jax
```

## Dataset

The default distillation recipe uses recaptioned CC12M WebDataset shards. Each sample must contain an image under `jpg` or `png` and a caption under `txt`. Point `CC12M_ROOT` in [`settings.py`](settings.py) at the shard pattern, for example:

```python
CC12M_ROOT = "/path/to/cc12m/{00000..01096}.tar"
# or
CC12M_ROOT = "gs://your-bucket/data/cc12m/{00000..01096}.tar"
```

## Distillation

Run the B/16 Mean Flow distillation recipe:

```bash
bash scripts/train.sh distill_b16 \
  --config configs/load_config.py:distill_b16 \
  --workdir /path/to/runs/minit2i-b16-mf \
  --load_pt_from /path/to/checkpoints/MiniT2I-B-16-jax
```

The config uses a four-step sampler, `cfg_scale: 12.0`, and the guidance interval `[0.2, 0.8]`, matching the released MiniT2I-B/16-MF recipe. Checkpoints are saved directly under `workdir` at `training.checkpoint_per_step` and at the final step.

## Resume

To resume a distillation run, pass the student checkpoint with `--load_from`. Keep `--load_pt_from` only if you also want to reload/verify the teacher path; the resumed checkpoint already contains the frozen teacher subtree.

```bash
bash scripts/train.sh distill_b16 \
  --config configs/load_config.py:distill_b16 \
  --workdir /path/to/runs/minit2i-b16-mf-resume \
  --load_from /path/to/runs/minit2i-b16-mf/checkpoint_10000
```

`load_from` may point either to a Flax checkpoint directory (starting with `checkpoint_`), or to a _parent directory_ containing Flax checkpoints. If a parent directory is given, the latest checkpoint_* under that directory is restored.

## Evaluation

Evaluate or sample from the released Mean Flow checkpoint:

```bash
bash scripts/eval.sh eval_distill_b16 \
  --config configs/load_config.py:eval_distill_b16 \
  --workdir /path/to/runs/minit2i-b16-mf-eval \
  --load_from /path/to/checkpoints/MiniT2I-B-16-MF-jax
```

The checked-in eval config enables GenEval and DPG-Bench and sets `eval_show_sample: True`. To generate visualization samples only, disable the benchmark blocks:

```yaml
eval:
  geneval:
    enable: False
  dpgbench:
    enable: False
```

FID is still wired for convenience when `eval.mscoco.enable` or `eval.mjhq.enable` is true, but the standard Mean Flow release sanity checks focus on GenEval and DPG-Bench.

## Config Aliases

[`configs/load_config.py`](configs/load_config.py) accepts either an alias or a YAML path:

| Alias | YAML |
| --- | --- |
| `distill`, `distill_b16` | `configs/distill_config_b16.yml` |
| `eval`, `eval_distill`, `eval_distill_b16` | `configs/eval_distill_config_b16.yml` |

## Citation

```bibtex
@misc{minit2i2026,
  title  = {MiniT2I: A Minimalist Baseline for Text-to-Image Generation},
  author = {Wang, Xianbang and Zhao, Hanhong and Lu, Yiyang and Zhou, Kangyang and Ma, Linrui and He, Kaiming},
  year   = {2026},
  url    = {https://peppaking8.github.io/#/post/minit2i}
}
```
