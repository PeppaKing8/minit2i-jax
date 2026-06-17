"""Convert FreedomIntelligence/ShareGPT-4o-Image into WebDataset shards."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image


def png_to_jpg(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


def add_sample(tar: tarfile.TarFile, key: str, jpg_bytes: bytes, caption: str) -> None:
    image_info = tarfile.TarInfo(f"{key}.jpg")
    image_info.size = len(jpg_bytes)
    tar.addfile(image_info, io.BytesIO(jpg_bytes))

    text_bytes = (caption.strip() + "\n").encode("utf-8")
    text_info = tarfile.TarInfo(f"{key}.txt")
    text_info.size = len(text_bytes)
    tar.addfile(text_info, io.BytesIO(text_bytes))


def upload(path: Path, bucket: str, *, delete_after_upload: bool) -> None:
    subprocess.run(["gcloud", "storage", "cp", "-n", str(path), bucket.rstrip("/") + "/"], check=True)
    if delete_after_upload:
        path.unlink()


def load_captions(args: argparse.Namespace) -> dict[int, str]:
    from transformers import AutoTokenizer

    meta_path = Path(
        hf_hub_download(
            args.repo_id,
            args.metadata_file,
            repo_type="dataset",
            local_dir=args.meta_dir,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data = json.loads(meta_path.read_text())
    captions: dict[int, str] = {}
    for item in data:
        output_image = item.get("output_image", "")
        if not (output_image.startswith("image/") and output_image.endswith(".png")):
            continue
        stem = output_image[len("image/") : -len(".png")]
        if not stem.isdigit():
            continue
        caption = item.get("input_prompt", "")
        tokenized = tokenizer(caption, return_tensors="pt", truncation=False)
        if tokenized.input_ids.shape[1] <= args.max_tokens:
            captions[int(stem)] = caption
    return captions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="FreedomIntelligence/ShareGPT-4o-Image")
    parser.add_argument("--metadata-file", default="text_to_image.json")
    parser.add_argument("--tar-pattern", default="text_to_image_part_{idx}.tar")
    parser.add_argument("--meta-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gcs-bucket", default="")
    parser.add_argument("--delete-after-upload", action="store_true")
    parser.add_argument("--shard-size", type=int, default=3000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--tokenizer", default="google/flan-t5-large")
    parser.add_argument("--max-parts", type=int, default=0)
    args = parser.parse_args()

    meta_dir = Path(args.meta_dir)
    output_dir = Path(args.output_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    captions = load_captions(args)
    print(f"Kept {len(captions)} captions after token filtering.")

    current_tar: tarfile.TarFile | None = None
    current_path: Path | None = None
    shard_idx = 0
    in_shard = 0
    total = 0
    errors: list[dict[str, object]] = []

    def close_current() -> None:
        nonlocal current_tar, current_path, shard_idx, in_shard
        if current_tar is None or current_path is None:
            return
        current_tar.close()
        print(f"Wrote {current_path}")
        if args.gcs_bucket:
            upload(current_path, args.gcs_bucket, delete_after_upload=args.delete_after_upload)
        current_tar = None
        current_path = None
        shard_idx += 1
        in_shard = 0

    part_idx = 0
    while True:
        if args.max_parts > 0 and part_idx >= args.max_parts:
            break
        filename = args.tar_pattern.format(idx=part_idx)
        try:
            part_path = Path(
                hf_hub_download(
                    args.repo_id,
                    filename,
                    repo_type="dataset",
                    local_dir=meta_dir,
                )
            )
        except Exception as exc:
            print(f"Stopping at part {part_idx}: {exc}")
            break

        print(f"Processing {filename}")
        with tarfile.open(part_path, "r:*") as source_tar:
            for member in source_tar:
                if not member.isfile() or not member.name.endswith(".png"):
                    continue
                stem = member.name[len("image/") : -len(".png")] if member.name.startswith("image/") else ""
                if not stem.isdigit():
                    continue
                image_id = int(stem)
                caption = captions.get(image_id)
                if not caption:
                    errors.append({"id": image_id, "reason": "no_caption"})
                    continue
                fileobj = source_tar.extractfile(member)
                if fileobj is None:
                    errors.append({"id": image_id, "reason": "no_file"})
                    continue
                try:
                    jpg_bytes = png_to_jpg(fileobj.read())
                except Exception as exc:
                    errors.append({"id": image_id, "reason": f"image_convert:{exc}"})
                    continue
                if current_tar is None:
                    current_path = output_dir / f"shard-{shard_idx:03d}.tar"
                    current_tar = tarfile.open(current_path, "w")
                add_sample(current_tar, str(image_id), jpg_bytes, caption)
                in_shard += 1
                total += 1
                if in_shard >= args.shard_size:
                    close_current()
        try:
            os.remove(part_path)
        except OSError:
            pass
        part_idx += 1

    close_current()
    if errors:
        err_path = output_dir / "errors.jsonl"
        with err_path.open("w") as f:
            for error in errors:
                f.write(json.dumps(error) + "\n")
        print(f"Wrote {len(errors)} errors to {err_path}")
    print(f"Done. Wrote {total} samples.")


if __name__ == "__main__":
    main()
