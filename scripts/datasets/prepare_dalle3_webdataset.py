"""Convert OpenDatasets/dalle-3-dataset into WebDataset shards."""

from __future__ import annotations

import argparse
import ast
import base64
import io
import os
import subprocess
import tarfile
from math import ceil
from pathlib import Path

from huggingface_hub import snapshot_download


def coerce_image_bytes(raw, base_dir: Path) -> bytes | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        if raw.get("bytes") is not None:
            return coerce_image_bytes(raw["bytes"], base_dir)
        if raw.get("path"):
            path = base_dir / raw["path"]
            return path.read_bytes() if path.exists() else None
        return None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw)
    if isinstance(raw, str):
        if raw.startswith("data:image"):
            try:
                return base64.b64decode(raw.split(",", 1)[1], validate=False)
            except Exception:
                return None
        try:
            return base64.b64decode(raw, validate=True)
        except Exception:
            pass
        try:
            literal = ast.literal_eval(raw)
            if isinstance(literal, (list, tuple)):
                return bytes(literal)
        except Exception:
            pass
    if isinstance(raw, (list, tuple)):
        try:
            return bytes(raw)
        except Exception:
            return None
    return None


def write_pair(tar: tarfile.TarFile, key: str, image_bytes: bytes, caption: str) -> None:
    image_info = tarfile.TarInfo(f"{key}.jpg")
    image_info.size = len(image_bytes)
    tar.addfile(image_info, io.BytesIO(image_bytes))

    text_bytes = (caption.strip() + "\n").encode("utf-8")
    text_info = tarfile.TarInfo(f"{key}.txt")
    text_info.size = len(text_bytes)
    tar.addfile(text_info, io.BytesIO(text_bytes))


def upload(path: Path, bucket: str, *, delete_after_upload: bool) -> None:
    subprocess.run(["gcloud", "storage", "cp", "-n", str(path), bucket.rstrip("/") + "/"], check=True)
    if delete_after_upload:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="OpenDatasets/dalle-3-dataset")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gcs-bucket", default="")
    parser.add_argument("--delete-after-upload", action="store_true")
    parser.add_argument("--parquets-per-shard", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-shards", type=int, default=0)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(
        snapshot_download(
            args.repo_id,
            repo_type="dataset",
            local_dir=raw_dir,
        )
    )
    parquet_dir = snapshot_path / "data"
    parquets = sorted(parquet_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found under {parquet_dir}")

    import pyarrow.parquet as pq

    num_shards = ceil(len(parquets) / args.parquets_per_shard)
    if args.max_shards > 0:
        num_shards = min(num_shards, args.max_shards)

    errors: list[str] = []
    for shard_idx in range(num_shards):
        shard_parquets = parquets[
            shard_idx * args.parquets_per_shard : (shard_idx + 1) * args.parquets_per_shard
        ]
        shard_path = output_dir / f"shard-{shard_idx:03d}.tar"
        written = 0
        with tarfile.open(shard_path, "w") as tar:
            global_idx = shard_idx * 1_000_000
            for parquet_path in shard_parquets:
                pf = pq.ParquetFile(parquet_path)
                row_offset = 0
                for batch in pf.iter_batches(batch_size=args.batch_size):
                    table = batch.to_pandas()
                    for local_offset, row in enumerate(table.itertuples(index=False)):
                        image_bytes = coerce_image_bytes(row.image, snapshot_path)
                        if not image_bytes:
                            errors.append(f"{parquet_path.name}:{row_offset + local_offset}:image")
                            continue
                        caption = getattr(row, "caption", "") or ""
                        key = getattr(row, "image_hash", None) or f"{global_idx:012d}"
                        write_pair(tar, str(key), image_bytes, caption)
                        global_idx += 1
                        written += 1
                    row_offset += len(table)
        print(f"Wrote {written} samples to {shard_path}")
        if args.gcs_bucket:
            upload(shard_path, args.gcs_bucket, delete_after_upload=args.delete_after_upload)

    if errors:
        err_path = output_dir / "errors.txt"
        err_path.write_text("\n".join(errors) + "\n")
        print(f"Wrote {len(errors)} errors to {err_path}")


if __name__ == "__main__":
    main()
