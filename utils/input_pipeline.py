"""WebDataset-based training input pipeline with LLM-side tokenization."""

import random
from functools import partial
import io
import os
import tarfile
import warnings
import jax
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms

from utils.logging_util import log_for_0

try:
    import webdataset as wds
    try:
        from webdataset.filters import RandomMix
    except ImportError:  # fallback to attribute on wds if available
        RandomMix = getattr(wds, "RandomMix", None)
except ImportError:
    wds = None
    RandomMix = None

import importlib
import fsspec

# WebDataset uses its own gopen dispatch table; the default handler for
# `gs://` shells out to `gsutil`, which is broken on some TPU images
# (pyOpenSSL/OpenSSL mismatch).  We override the handler to go through
# fsspec+gcsfs so credentials from GOOGLE_APPLICATION_CREDENTIALS still
# work but without invoking the snap-installed gsutil.
def _register_gcsfs():
    if wds is None:
        return
    gopen_module = importlib.import_module("webdataset.gopen")

    def gopen_gcsfs(url, mode="rb", bufsize=8192, **kw):
        # fsspec.open uses gcsfs under the hood for gs:// and handles auth.
        return fsspec.open(url, mode=mode).open()

    gopen_module.gopen_schemes["gs"] = gopen_gcsfs


_register_gcsfs()

def _get_image_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def prepare_batch_data(batch):
    """
    Reformat a input batch from PyTorch Dataloader.
    """
    
    # batch: {'pixel_values': <class 'torch.Tensor'>, 'input_ids': <class 'torch.Tensor'>, 'attention_mask': <class 'torch.Tensor'>, '__key__': <class 'list'>}
    batch = {k: v for k, v in batch.items() if k != '__key__'}
    batch = {k: v.numpy() for k, v in batch.items()}
    batch['pixel_values'] = batch['pixel_values'].transpose(0, 2, 3, 1)  # to NHWC
    
    return batch


def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


IMAGE_KEYS = ['jpg', 'png']
TEXT_KEY = 'txt'


class LocalTarDataset(IterableDataset):
    """Small local-tar fallback used when WebDataset is not installed.

    This keeps release smoke tests runnable on TPU images that skip
    `requirements.txt`. For real training, install the pinned `webdataset`
    dependency and use the streaming pipeline below.
    """

    def __init__(self, roots, dataset_cfg, llm, rank):
        if isinstance(roots, str):
            roots = [roots]
        self.roots = list(roots)
        self.dataset_cfg = dataset_cfg
        self.llm = llm
        self.rank = rank

    def _load_root(self, root):
        if not os.path.exists(root):
            raise FileNotFoundError(
                f"LocalTarDataset fallback only supports existing local tar files; "
                f"got {root!r}. Install webdataset for remote URLs or shard patterns."
            )
        groups = {}
        with tarfile.open(root, "r:*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                stem, ext = os.path.splitext(member.name)
                key = ext.lstrip(".").lower()
                if key not in (*IMAGE_KEYS, TEXT_KEY):
                    continue
                fp = tar.extractfile(member)
                if fp is None:
                    continue
                groups.setdefault(stem, {})[key] = fp.read()

        samples = []
        for sample in groups.values():
            image_key = next((key for key in IMAGE_KEYS if key in sample), None)
            if image_key is None or TEXT_KEY not in sample:
                continue
            image = Image.open(io.BytesIO(sample[image_key])).convert("RGB")
            samples.append({
                image_key: image,
                TEXT_KEY: sample[TEXT_KEY].decode("utf-8", errors="replace"),
            })
        if not samples:
            raise ValueError(f"No image/text pairs found in local tar {root!r}.")
        return samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(114 + self.rank * 514 + worker_id)
        samples = []
        for root in self.roots:
            samples.extend(self._load_root(root))
        while True:
            rng.shuffle(samples)
            for sample in samples:
                yield llm_preprocess_fn(self.dataset_cfg, self.llm, sample)

def llm_preprocess_fn(dataset_cfg, llm, sample):
    transform = _get_image_transform(dataset_cfg.image_size)
    image = None
    for key in IMAGE_KEYS:
        if key in sample:
            image = sample[key]
            break

    assert image is not None, f"No image found in sample with keys {sample.keys()}"
    pixel_values = transform(image)

    caption = sample.get(TEXT_KEY, "")
    input_ids, attention_mask = llm.tokenize_single(caption)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def make_stop_after_n_errors(max_errors=50):
    """Skip sporadic bad samples; stop after too many errors."""
    count = [0]

    def handler(exn):
        count[0] += 1
        if count[0] >= max_errors:
            raise exn
        warnings.warn(f"Ignoring error ({count[0]}/{max_errors}): {exn}", UserWarning, stacklevel=2)
        return True

    return handler

def create_split(dataset_cfg, batch_size, llm):
    """Build the WebDataset training dataloader.

    Mixes the configured shard roots with the given weights, tokenizes captions
    via `llm`, and yields batches of (pixel_values, input_ids, attention_mask).
    """
    rank = jax.process_index()
    root = dataset_cfg.root

    if wds is None:
        log_for_0(
            "webdataset is not installed; using LocalTarDataset fallback. "
            "Install requirements.txt for full streaming WebDataset support."
        )
        loader = DataLoader(
            LocalTarDataset(root, dataset_cfg, llm, rank),
            batch_size=batch_size,
            drop_last=True,
            worker_init_fn=partial(worker_init_fn, rank=rank),
            num_workers=dataset_cfg.num_workers,
            prefetch_factor=(
                dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None
            ),
            pin_memory=dataset_cfg.pin_memory,
            persistent_workers=dataset_cfg.num_workers > 0,
        )
        log_for_0("PyTorch DataLoader wrapped with LocalTarDataset fallback.")
        return loader

    # Rank+worker unique seed so different (process, worker) pairs read
    # different shards.
    def custom_worker_seed():
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        return rank * 10000 + worker_id  # unique seed per (process, worker)
    
    def make_wds(url, seed_offset=0):
        log_for_0(f"Creating ResampledShards for {url}...")
        shards = wds.ResampledShards(
            url,
            deterministic=False,
            worker_seed=custom_worker_seed,
        )
        log_for_0("ResampledShards created.")
        log_for_0("Wrapping WebDataset data pipeline...")
        ds = wds.DataPipeline(
            shards,
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=make_stop_after_n_errors(500)),
            wds.shuffle(1000, rng=random.Random(114 + rank * 514 + seed_offset)),
            wds.decode("pil", handler=make_stop_after_n_errors(500)),
            wds.map(
                partial(
                    llm_preprocess_fn,
                    dataset_cfg,
                    llm,
                ),
                handler=make_stop_after_n_errors(500),
            ),
            handler=make_stop_after_n_errors(500),
        )
        log_for_0("WebDataset data pipeline wrapped.")
        return ds

    if isinstance(root, (list, tuple)) and len(root) == 1:
        root = root[0]

    if isinstance(root, (list, tuple)):
        weights = getattr(dataset_cfg, "mix_weights", [])
        if not weights or len(weights) != len(root):
            weights = [1.0] * len(root)
        datasets = [make_wds(u, i) for i, u in enumerate(root)]
        mix_cls = RandomMix if RandomMix is not None else getattr(wds, "RandomMix")
        ds = mix_cls(datasets, weights)
        log_for_0(f'Mixed datasets with weights {weights}')
    else:
        ds = make_wds(root)
        log_for_0(f'Dataset at {root} created.')
    log_for_0("Wrapping stream in PyTorch DataLoader...")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=True,
        worker_init_fn=partial(worker_init_fn, rank=rank),
        num_workers=dataset_cfg.num_workers,
        prefetch_factor=(
            dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None
        ),
        pin_memory=dataset_cfg.pin_memory,
        persistent_workers=dataset_cfg.num_workers > 0,
    )
    log_for_0("PyTorch DataLoader wrapped.")
    return loader
