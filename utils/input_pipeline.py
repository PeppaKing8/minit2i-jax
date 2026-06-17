"""WebDataset-based training input pipeline with LLM-side tokenization."""

import io
import importlib
import random
import re
import tarfile
from functools import partial
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


def _flatten_roots(roots):
    if isinstance(roots, str):
        return [roots]
    flat = []
    for root in roots:
        flat.extend(_flatten_roots(root))
    return flat


def _expand_numeric_braces(path):
    match = re.search(r"\{(\d+)\.\.(\d+)\}", path)
    if match is None:
        return [path]
    start_s, end_s = match.groups()
    width = max(len(start_s), len(end_s))
    start, end = int(start_s), int(end_s)
    step = 1 if end >= start else -1
    expanded = []
    for value in range(start, end + step, step):
        expanded.append(path[:match.start()] + f"{value:0{width}d}" + path[match.end():])
    return expanded


class TarShardDataset(IterableDataset):
    """Dependency-light tar-shard reader used when `webdataset` is unavailable."""

    def __init__(self, roots, dataset_cfg, llm, rank):
        shards = []
        for root in _flatten_roots(roots):
            shards.extend(_expand_numeric_braces(root))
        if not shards:
            raise ValueError("No tar shards configured for TarShardDataset.")
        self.shards = shards
        self.dataset_cfg = dataset_cfg
        self.llm = llm
        self.rank = rank
        self.process_count = jax.process_count()

    def _iter_samples_from_shard(self, shard):
        groups = {}
        with fsspec.open(shard, "rb").open() as fp:
            with tarfile.open(fileobj=fp, mode="r|*") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    stem, ext = member.name.rsplit(".", 1) if "." in member.name else (member.name, "")
                    key = ext.lower()
                    if key not in (*IMAGE_KEYS, TEXT_KEY):
                        continue
                    fileobj = tar.extractfile(member)
                    if fileobj is None:
                        continue
                    groups.setdefault(stem, {})[key] = fileobj.read()

        for sample in groups.values():
            image_key = next((key for key in IMAGE_KEYS if key in sample), None)
            if image_key is None or TEXT_KEY not in sample:
                continue
            image = Image.open(io.BytesIO(sample[image_key])).convert("RGB")
            yield {
                image_key: image,
                TEXT_KEY: sample[TEXT_KEY].decode("utf-8", errors="replace"),
            }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        world_workers = max(self.process_count * num_workers, 1)
        worker_rank = self.rank * num_workers + worker_id
        rng = random.Random(114 + worker_rank)

        while True:
            shards = self.shards[:]
            rng.shuffle(shards)
            for shard in shards[worker_rank::world_workers]:
                try:
                    samples = list(self._iter_samples_from_shard(shard))
                except Exception as exc:
                    warnings.warn(f"Skipping shard {shard}: {exc}", UserWarning, stacklevel=2)
                    continue
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
            "webdataset is not installed; using TarShardDataset fallback. "
            "Install requirements.txt for the faster WebDataset pipeline."
        )
        loader = DataLoader(
            TarShardDataset(root, dataset_cfg, llm, rank),
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
        log_for_0("PyTorch DataLoader wrapped with TarShardDataset fallback.")
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
