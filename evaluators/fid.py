import json
import os
import time
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.experimental import multihost_utils
from PIL import Image
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from torchvision import transforms

from external.jax_fid import inception, resize
from external.jax_fid.fid import compute_frechet_distance
from external.jax_fid.cvt import load_all as experimental_load_all
from flax.jax_utils import replicate as R

from utils import sample_util
from utils.logging_util import log_for_0

DEFAULT_DATASET_CACHE = "/dev/shm"
IMAGE_IDENTIFIERS = {".jpg", ".jpeg", ".png"}

# mjhq dataset info
MJHQ_ARCHIVE_NAME = "mjhq30k_imgs.zip"
MJHQ_REPO_ID = "playgroundai/MJHQ-30K"
MJHQ_CATEGORIES = ["animals", "art", "fashion", "food", "indoor", "landscape", "logo", "people", "plants", "vehicles"]
MJHQ_CAPTION_FILE = ""

# mscoco dataset info
MSCOCO_REPO_ID = "sayakpaul/coco-30-val-2014"
MSCOCO_CAPTION_FILE = ""

def build_jax_inception(batch_size=200):
    """
    Build InceptionV3 model that always returns all features.
    
    Args:
        batch_size: Batch size for compilation
        
    Returns:
        Dictionary with model parameters and compiled function
    """
    log_for_0("Initializing Extended InceptionV3...")
    model = inception.InceptionV3(
        pretrained=True,
        include_head=True,  # Need head for logits
        transform_input=False  # Already normalized in resize.forward
    )
    
    inception_params = experimental_load_all()
    inception_params = R(inception_params)
    
    log_for_0("Initialized Extended InceptionV3")
    multihost_utils.sync_global_devices("inception init finished")
    
    # Create a single function that always returns all features
    def inception_apply(params, x):
        return model.apply(params, x, train=False)
    
    # JIT compile the function
    
    # Compile for the expected batch size
    fake_x = jnp.zeros((jax.local_device_count(), batch_size, 299, 299, 3), dtype=jnp.float32)
    log_for_0('Start compiling inception function...')
    t_start = time.time()
    
    # Trigger compilation
    # _ = inception_fn(inception_params, fake_x)
    inception_fn = jax.pmap(inception_apply)
    inception_fn = inception_fn.lower(inception_params, fake_x).compile()
    
    log_for_0(f'End compiling: {(time.time() - t_start):.4f} seconds.')
    
    inception_net = {
        "params": inception_params, 
        "fn": inception_fn,
        "model": model,
        "per_device_batch_size": batch_size,
    }
    return inception_net


def get_reference(cache_path):
    # Load ref_mu and ref_sigma from npz file
    assert os.path.exists(cache_path), f"Cache file must exist: {cache_path}"

    log_for_0(f"Loading ref_mu and ref_sigma from {cache_path}")
    if jax.process_index() == 0:
        os.system('md5sum ' + cache_path)

    ref = {}
    with np.load(cache_path) as data:
        if "ref_mu" in data:
            ref["mu"], ref["sigma"] = data["ref_mu"], data["ref_sigma"]
        elif "mu" in data and "sigma" in data:
            ref["mu"], ref["sigma"] = data["mu"], data["sigma"]
        else:
            raise ValueError(f"Invalid FID stats file: {cache_path}. Its keys are: {list(data.keys())}")

    return ref

LDC = jax.local_device_count()
def revert_pmap_shape(x):
    return x.reshape((-1, *x.shape[2:]))

def _batched_images(images: Iterable[np.ndarray] | np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    if isinstance(images, np.ndarray):
        for i in range(0, len(images), batch_size):
            yield images[i:i + batch_size]
        return

    batch = []
    for img in images:
        if img is None:
            continue
        batch.append(np.asarray(img))
        if len(batch) == batch_size:
            # assert all images in the batch have the same shape
            shapes = [img.shape for img in batch]
            # find the unique shapes
            unique_shapes = set(shapes)
            assert len(unique_shapes) == 1, f"All images in the batch must have the same shape, but got: {unique_shapes}"
            yield np.stack(batch)
            batch = []
    if batch:
        yield np.stack(batch)

def compute_stats(
    images,
    inception_net,
    *,
    batch_size: int | None = None,
    fid_samples: int | None = 30000,
    gather: bool = True,
):
    """Compute FID statistics (mu, sigma) for a set of images."""
    inception_fn = inception_net["fn"]
    inception_params = inception_net["params"]
    per_device_batch = batch_size or inception_net.get("per_device_batch_size", 200)
    full_batch_size = per_device_batch * LDC

    feats = []
    seen = 0
    for batch in _batched_images(images, full_batch_size):
        if fid_samples is not None and seen >= fid_samples:
            break

        batch = np.asarray(batch, dtype=np.uint8)
        current_batch = batch.shape[0]
        if fid_samples is not None:
            current_batch = min(current_batch, fid_samples - seen)
            batch = batch[:current_batch]
        if current_batch == 0:
            break

        if current_batch < full_batch_size:
            padding_shape = (full_batch_size - current_batch,) + batch.shape[1:]
            batch = np.concatenate([batch, np.zeros(padding_shape, dtype=np.uint8)], axis=0)

        x = torch.from_numpy(batch.astype(np.float32).transpose(0, 3, 1, 2))
        x = resize.forward(x)
        x = x.numpy().transpose(0, 2, 3, 1)
        x = x.reshape((LDC, -1, *x.shape[1:]))

        pooled_features, _, _ = inception_fn(
            inception_params,
            jax.lax.stop_gradient(x)
        )
        pooled_features = revert_pmap_shape(pooled_features)
        feats.append(pooled_features[:current_batch])
        seen += current_batch
        log_for_0(f"Computed features for {seen} images")

    if not feats:
        raise ValueError("No images provided to compute FID statistics")

    np_feats = jnp.concatenate(feats, axis=0)
    if gather:
        all_feats = multihost_utils.process_allgather(np_feats)
        all_feats = all_feats.reshape(-1, np_feats.shape[-1])
        all_feats = jax.device_get(all_feats)
    else:
        all_feats = np.array(jax.device_get(np_feats))
        
    log_for_0(f"Features gathered. Before truncation: {all_feats.shape}")

    if fid_samples is not None:
        all_feats = all_feats[:fid_samples]

    feats_64 = all_feats.astype(np.float64)
    mu = np.mean(feats_64, axis=0)
    sigma = np.cov(feats_64, rowvar=False)
    return mu, sigma

def _find_category_root(root: Path, categories: Sequence[str]) -> Path:
    """Find the directory containing the expected MJHQ category folders."""
    if root.is_file():
        root = root.parent
    for current_root, dirnames, _ in os.walk(root):
        if all(cat in dirnames for cat in categories):
            return Path(current_root)
    raise FileNotFoundError(f"Did not find required category folders {categories} under {root}")

def _prepare_mjhq(cache_dir: str) -> Path:
    cache_dir = cache_dir or DEFAULT_DATASET_CACHE
    archive_path = hf_hub_download(
        repo_id=MJHQ_REPO_ID,
        repo_type="dataset",
        filename=MJHQ_ARCHIVE_NAME,
        local_dir=cache_dir,
    )
    extract_dir = Path(cache_dir)
    extracted_flag = extract_dir / f".{MJHQ_ARCHIVE_NAME}.extracted"
    if not extracted_flag.exists():
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
        extracted_flag.touch()

    return _find_category_root(extract_dir, MJHQ_CATEGORIES)

def _prepare_mscoco(cache_dir: str):
    cache_dir = cache_dir or DEFAULT_DATASET_CACHE
    dataset = load_dataset(MSCOCO_REPO_ID, split="train", cache_dir=cache_dir)
    return dataset

def get_fid_stats(
    dataset: str,
    cache_path: str,
    *,
    inception_net=None,
    cache_dir: str = DEFAULT_DATASET_CACHE,
    batch_size: int | None = None,
    image_size: int = 512,
):
    """Load or compute dataset FID stats (mu, sigma)."""
    cache_file = Path(cache_path)
    if cache_file.exists():
        log_for_0(f"Find FID stats from {cache_file}.")
        data = np.load(cache_file)
        return data["mu"], data["sigma"]
    
    img_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        (transforms.CenterCrop(image_size)),
    ])

    if jax.process_index() == 0:
        assert inception_net is not None, \
            f"{dataset} FID stats not found at {cache_file}, so inception_net must be provided to compute them."
        if dataset.lower() == "mjhq":
            image_root = _prepare_mjhq(cache_dir)
            log_for_0(f"Found prepared MJHQ dataset at {image_root}.")
            image_paths = list(image_root.rglob("*"))
            def _iter_images():
                for img_path in sorted(image_paths):
                    if img_path.suffix.lower() not in IMAGE_IDENTIFIERS:
                        continue
                    with Image.open(img_path) as img:
                        yield np.asarray(img_transform(img.convert("RGB")))
        elif dataset.lower() == "mscoco":
            ds = _prepare_mscoco(cache_dir)
            log_for_0(f"Found prepared MSCOCO dataset with {len(ds)} samples.")
            def _iter_images():
                for item in ds:
                    img: Image.Image = item['image']
                    img = img_transform(img.convert("RGB"))
                    yield np.asarray(img)
        else:
            raise ValueError(f"Unsupported dataset for FID stats: {dataset}")

        log_for_0(f"Computing FID stats for dataset {dataset}...")
        mu, sigma = compute_stats(
            _iter_images(),
            inception_net=inception_net,
            batch_size=batch_size,
            fid_samples=None,
            gather=False,
        )
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, mu=mu, sigma=sigma)
        log_for_0(f"Saved FID stats to {cache_file}.")

    multihost_utils.sync_global_devices("fid_stats_ready")
    if not cache_file.exists():
        raise FileNotFoundError(f"FID stats not found at {cache_file}")

    data = np.load(cache_file)
    return data["mu"], data["sigma"]

def compute_fid(
    samples: np.ndarray,
    dataset: str,
    *,
    cache_path: str,
    inception_net,
    cache_dir: str = DEFAULT_DATASET_CACHE,
    fid_samples: int | None = 30000,
    batch_size: int | None = None,
    image_size: int = 512,
):
    """Compute FID between generated samples and the reference dataset."""
    mu_fake, sigma_fake = compute_stats(
        samples,
        inception_net,
        batch_size=batch_size,
        fid_samples=fid_samples,
    )
    mu_real, sigma_real = get_fid_stats(
        dataset,
        cache_path,
        inception_net=inception_net,
        cache_dir=cache_dir,
        batch_size=batch_size,
        image_size=image_size,
    )
    return compute_frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake)


def get_fid_evaluator(workdir, config, p_sample_step, run_p_sample_step, mesh_bundle=None):
    """Build a FID evaluator over the FID-style datasets enabled in config.

    Returns an `evaluator(params, step, writer, cfg_scale, descriptor)` closure
    that emits one `FID/<dataset>/<descriptor>/cfg<x>` metric per enabled
    target (mjhq, mscoco). Returns None if no FID datasets are enabled.
    InceptionV3 weights load once at construction time and are reused across
    eval rounds (multiple EMA params / cfg scales in eval_only).
    """
    del mesh_bundle  # Inception runs replicated under pmap, not pjit.

    eval_targets = []
    if getattr(config.eval.mjhq, "enable", False):
        eval_targets.append(("mjhq", config.eval.mjhq.stats_cache, config.eval.mjhq.caption_file or MJHQ_CAPTION_FILE))
    if getattr(config.eval.mscoco, "enable", False):
        eval_targets.append(("mscoco", config.eval.mscoco.stats_cache, config.eval.mscoco.caption_file or MSCOCO_CAPTION_FILE))
    if not eval_targets:
        return None

    inception_net = build_jax_inception()
    num_samples = config.eval.num_samples

    def evaluator(params, step, writer, cfg_scale=1.0, descriptor=""):
        del step, writer  # caller writes the merged metrics dict.
        metrics = {}
        vis_samples = None
        for dataset, cache_path, caption_file in eval_targets:
            if not caption_file:
                raise ValueError(f"FID for {dataset} requires eval.{dataset}.caption_file to be set.")
            log_for_0(f"Loading captions for {dataset} from {caption_file} ...")
            with open(caption_file, "r") as f:
                captions_raw = json.load(f)
                assert isinstance(captions_raw, dict), \
                    f"Unsupported caption file format for dataset {dataset}: {type(captions_raw)}"
                captions = list(captions_raw.values())
            log_for_0(f"Loaded {len(captions)} captions for dataset={dataset}.")

            samples_all = sample_util.generate_fid_samples(
                params, workdir, config, p_sample_step, run_p_sample_step,
                prompts=captions, cfg_scale=cfg_scale,
            )
            if vis_samples is None:
                vis_samples = samples_all
            log_for_0(f"Generated samples shape: {samples_all.shape}")

            fid_value = compute_fid(
                samples_all,
                dataset,
                cache_path=cache_path,
                inception_net=inception_net,
                fid_samples=num_samples,
                image_size=config.dataset.image_size,
            )
            des = f"{dataset}/{descriptor}/cfg{cfg_scale:.1f}"
            log_for_0(f"FID/{des} at {num_samples} samples: {fid_value}")
            metrics[f"FID/{des}"] = fid_value
        return metrics, vis_samples

    return evaluator
