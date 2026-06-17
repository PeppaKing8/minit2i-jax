"""Convert CC12M recaption JSONL into an img2dataset parquet input.

Expected source:
  CaptionEmporium/conceptual-captions-cc12m-llavanext train.jsonl.gz

Output columns:
  url, caption, caption_llava_short, key

`img2dataset` reads the `url` and `caption` columns to produce WebDataset
shards containing `{key}.jpg` and `{key}.txt` pairs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to train.jsonl.gz.")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keep_columns = ["url", "caption_llava", "caption_llava_short", "key"]
    rename_map = {"caption_llava": "caption"}

    reader = pd.read_json(
        input_path,
        lines=True,
        compression="gzip",
        chunksize=args.chunksize,
    )
    first_chunk = True
    total_rows = 0
    for chunk in reader:
        if "status" in chunk.columns:
            chunk = chunk[chunk["status"] == "success"]
        missing = [col for col in keep_columns if col not in chunk.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        clean = chunk[keep_columns].rename(columns=rename_map)
        clean.to_parquet(
            output_path,
            index=False,
            engine="fastparquet",
            append=not first_chunk,
        )
        first_chunk = False
        total_rows += len(clean)
        print(f"Processed {total_rows} rows", end="\r", flush=True)
    print(f"\nWrote {total_rows} rows to {output_path}")


if __name__ == "__main__":
    main()
