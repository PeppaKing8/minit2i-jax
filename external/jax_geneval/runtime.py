from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

from .config import DetectorConfig
from .convert import convert_mask2former_detector_state_dict
from .detector import mask2former_detector_forward, mask2former_detector_instances


def resolve_checkpoint_path(
    path: str,
    *,
    cache_dir: str | None = None,
) -> str:
    """Resolve local or GCS checkpoint paths to a local file path."""

    if not path:
        return path
    if not path.startswith("gs://"):
        return str(Path(path).expanduser())

    cache_root = Path(cache_dir or "/tmp/jax_geneval_ckpts")
    cache_root.mkdir(parents=True, exist_ok=True)
    basename = os.path.basename(path.rstrip("/")) or "checkpoint"
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]
    local_path = cache_root / f"{digest}-{basename}"
    if local_path.exists():
        return str(local_path)

    tmp_path = local_path.with_name(f".{local_path.name}.tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()

    if shutil.which("gsutil"):
        cmd = ["gsutil", "cp", path, str(tmp_path)]
    elif shutil.which("gcloud"):
        cmd = ["gcloud", "storage", "cp", path, str(tmp_path)]
    else:
        raise RuntimeError(
            f"Cannot copy GCS checkpoint {path!r}: neither `gsutil` nor `gcloud` is available."
        )

    subprocess.run(cmd, check=True)
    os.replace(tmp_path, local_path)
    return str(local_path)


def load_params(checkpoint_path: str, *, cache_dir: str | None = None) -> dict[str, object]:
    if not checkpoint_path:
        raise ValueError(
            "Missing detector checkpoint. Pass a checkpoint path or set "
            "JAX_GENEVAL_DETECTOR_CKPT."
        )
    checkpoint_path = resolve_checkpoint_path(checkpoint_path, cache_dir=cache_dir)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return convert_mask2former_detector_state_dict(checkpoint["state_dict"])


def create_infer_fn(
    params: dict[str, object],
    *,
    detector_cfg: DetectorConfig,
    compile_mode: str,
    batch_size: int,
    mesh: Any | None = None,
):
    """Create a fixed-shape detector inference function.

    When ``mesh`` is provided, the compiled function expects global arrays
    sharded on the first dimension over every mesh axis. This is the mode used
    by text-jit online eval: host-local batches are converted to global arrays
    by the training code's mesh helpers, then detector outputs are scattered
    back to host-local arrays.
    """

    def forward(p, images):
        if not detector_cfg.device_instance_postprocess:
            return mask2former_detector_forward(images, p, detector_cfg=detector_cfg)
        return mask2former_detector_instances(images, p, detector_cfg=detector_cfg)

    # Resolve `auto`: prefer host-local pmap when there are multiple local
    # chips, else fall back to single-device jit. We deliberately do NOT
    # auto-pick pjit even with a global mesh, because cross-host pjit has
    # historically caused silent TPU aborts at the first Mask2Former forward
    # on multi-host pods (see comment in JaxGenevalEvaluator.__init__).
    if compile_mode == "auto":
        compile_mode = "pmap" if jax.local_device_count() > 1 else "jit"

    use_pjit = compile_mode == "pjit" or mesh is not None
    use_pmap = compile_mode == "pmap" and not use_pjit
    if use_pjit:
        devices = list(mesh.devices.flat) if mesh is not None else jax.devices()
        if batch_size % len(devices) != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by device_count={len(devices)} for pjit"
            )
        from jax.experimental.pjit import pjit
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        if mesh is None:
            mesh = Mesh(np.asarray(devices), ("data",))
            data_axes: str | tuple[str, ...] = "data"
        else:
            data_axes = tuple(mesh.axis_names)

        replicated = NamedSharding(mesh, P())
        image_sharding = NamedSharding(mesh, P(data_axes, None, None, None))
        if detector_cfg.device_instance_postprocess:
            out_sharding = (
                NamedSharding(mesh, P(data_axes, None)),
                NamedSharding(mesh, P(data_axes, None, None)),
                NamedSharding(mesh, P(data_axes, None, None, None)),
            )
        else:
            out_sharding = (
                NamedSharding(mesh, P(data_axes, None, None)),
                NamedSharding(mesh, P(data_axes, None, None, None)),
            )

        with mesh:
            params_dev = jax.device_put(params, replicated)
            compiled = pjit(
                forward,
                in_shardings=(replicated, image_sharding),
                out_shardings=out_sharding,
            )

        def infer(images):
            with mesh:
                if isinstance(images, jax.Array):
                    images_dev = images
                else:
                    images_dev = jax.device_put(jnp.asarray(images), image_sharding)
                return compiled(params_dev, images_dev)

        return infer, "pjit"

    if use_pmap:
        # Host-local data parallel: replicate Mask2Former params onto every
        # local chip, split the per-host batch evenly across them. No cross-
        # host collectives - this is safe even on the pods where pjit aborts.
        local_devices = jax.local_devices()
        local_dev_count = len(local_devices)
        if batch_size % local_dev_count != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by local_device_count="
                f"{local_dev_count} for pmap"
            )
        per_chip_batch = batch_size // local_dev_count
        params_repl = jax.device_put_replicated(params, local_devices)
        compiled = jax.pmap(forward)

        def infer(images):
            arr = jnp.asarray(images)
            arr = arr.reshape(local_dev_count, per_chip_batch, *arr.shape[1:])
            outs = compiled(params_repl, arr)
            # Bring to host and flatten the device axis so callers see a tuple
            # of (batch_size, ...) arrays matching the jit path. `np.asarray`
            # blocks on the pmap compute, so the caller's later
            # block_until_ready + device_get on the result become no-ops.
            return jax.tree_util.tree_map(
                lambda x: np.asarray(x).reshape(batch_size, *x.shape[2:]),
                outs,
            )

        return infer, "pmap"

    params_dev = jax.device_put(params)
    compiled = jax.jit(forward)

    def infer(images):
        return compiled(params_dev, jnp.asarray(images))

    return infer, "jit"
