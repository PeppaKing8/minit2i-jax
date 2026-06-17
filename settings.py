"""Local settings for datasets, checkpoint roots, evaluation assets, and logging.

Edit this file for your machine or cluster. The experiment configs in
`configs/` read these values through `configs/default.py`, so release configs
do not need to contain private filesystem paths or W&B entities.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# User settings: edit these paths for your machine or cluster.
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/path/to/dataset/root")
OUTPUT_ROOT = Path("/path/to/output/root")
CHECKPOINT_ROOT = Path("/path/to/checkpoint/root")
EVAL_ASSET_ROOT = Path("/path/to/evaluation/assets")

# ---------------------------------------------------------------------------
# Derived settings below. You usually do not need to edit anything here.
# ---------------------------------------------------------------------------

# Training datasets. Each shard should be a WebDataset tar with image key
# `jpg` or `png` and caption key `txt`.
CC12M_ROOT = str(DATA_ROOT / "cc12m" / "{00000..01096}.tar")
BLIP3_FT60K_ROOT = str(DATA_ROOT / "BLIP3o-60k")
DALLE3_ROOT = str(DATA_ROOT / "dalle3" / "shard-{000..013}.tar")
SHAREGPT4O_ROOT = str(DATA_ROOT / "gpt4oimg" / "shard-{000..013}.tar")

# FID assets.
MSCOCO_FID_STATS = str(EVAL_ASSET_ROOT / "coco" / "jax_mscoco_fid_stats_512.npz")
MSCOCO_CAPTION_FILE = str(EVAL_ASSET_ROOT / "coco" / "coco-30-val-2014-captions.json")

# GenEval assets.
GENEVAL_METADATA_FILE = str(EVAL_ASSET_ROOT / "geneval" / "evaluation_metadata.jsonl")
GENEVAL_DETECTOR_CHECKPOINT = str(
  EVAL_ASSET_ROOT / "geneval" / "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
)
GENEVAL_CLIP_CHECKPOINT = str(EVAL_ASSET_ROOT / "geneval" / "ViT-L-14.pt")
GENEVAL_CLIP_REPO = str(EVAL_ASSET_ROOT / "geneval" / "jax-clip")

# DPG-Bench assets.
DPG_BENCH_PROMPTS_DIR = str(EVAL_ASSET_ROOT / "dpg_bench" / "prompts")
DPG_BENCH_CSV = str(EVAL_ASSET_ROOT / "dpg_bench" / "dpg_bench.csv")
MPLUG_CHECKPOINT = str(EVAL_ASSET_ROOT / "modelscope" / "iic" / "mplug_visual-question-answering_coco_large_en")

# Logging.
USE_WANDB = False
USE_TB = False
WANDB_PROJECT = ""
WANDB_ENTITY = ""
WANDB_NOTES = ""
WANDB_TAGS = []
