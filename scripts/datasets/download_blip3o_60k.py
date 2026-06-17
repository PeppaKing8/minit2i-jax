"""Download BLIP3o-60K WebDataset shards from Hugging Face.

BLIP3o-60K is already released as WebDataset tar files containing `{key}.jpg`
and `{key}.txt` pairs, so this script only downloads and optionally uploads
those release shards.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "BLIP3o/BLIP3o-60k"
EXPECTED_FILES = (
    "dalle3.tar",
    "geneval_train.tar",
    "human_gestures.tar",
    "journeyDB.tar",
    "mscoco_human.tar",
    "object_1.tar",
    "object_2.tar",
    "occupation_1.tar",
    "occupation_2.tar",
    "text_1.tar",
    "text_2.tar",
)


def list_files(repo_type: str, token: str | None) -> list[str]:
    files = HfApi().list_repo_files(
        repo_id=REPO_ID,
        repo_type=repo_type,
        token=token or None,
    )
    by_base = {Path(path).name: path for path in files}
    missing = [name for name in EXPECTED_FILES if name not in by_base]
    if missing:
        raise FileNotFoundError(f"Missing expected files: {missing}")
    return [by_base[name] for name in EXPECTED_FILES]


def upload_to_gcs(path: Path, bucket: str, *, delete_after_upload: bool) -> None:
    dest = bucket.rstrip("/") + "/"
    subprocess.run(["gcloud", "storage", "cp", "-n", str(path), dest], check=True)
    if delete_after_upload:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--gcs-bucket", default="", help="Optional gs:// destination.")
    parser.add_argument("--delete-after-upload", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--token", default="", help="Optional Hugging Face token.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_files(repo_type=args.repo_type, token=args.token)
    if args.max_files > 0:
        files = files[: args.max_files]
    print(f"Downloading {len(files)} shard(s) from {REPO_ID}")
    for idx, repo_path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {repo_path}")
        local_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_path,
                repo_type=args.repo_type,
                token=args.token or None,
                local_dir=output_dir,
            )
        )
        if args.gcs_bucket:
            upload_to_gcs(
                local_path,
                args.gcs_bucket,
                delete_after_upload=args.delete_after_upload,
            )


if __name__ == "__main__":
    main()
