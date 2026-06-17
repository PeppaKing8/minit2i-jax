# Dataset Preparation

In this README, we introduce our recommended pipeline to install _all training datasets_ used in MiniT2I.

> MiniT2I expects flat **[WebDataset](https://github.com/webdataset/webdataset) tar shards**. Each sample should contain one image file (`.jpg` or `.png`) and one caption file (`.txt`) with the same key.

## CC12M Pretraining

First, download the recaptioned CC12M URL metadata:

```bash
hf download CaptionEmporium/conceptual-captions-cc12m-llavanext \
  --repo-type dataset \
  --local-dir /path/to/raw/cc12m-recaption
```

Then, run `prepare_cc12m_metadata.py` to convert the recaptioned CC12M JSONL release into a parquet file that `img2dataset` can read:

```bash
python scripts/datasets/prepare_cc12m_metadata.py \
  --input /path/to/train.jsonl.gz \
  --output /path/to/cc12m_llava.parquet
```

Then run `img2dataset` on the parquet file to download images and write WebDataset tar shards:

```bash
img2dataset \
  --url_list /path/to/work/cc12m_llava.parquet \
  --input_format parquet \
  --url_col url \
  --caption_col caption \
  --output_format webdataset \
  --output_folder /path/to/datasets/cc12m \
  --processes_count 16 \
  --thread_count 256 \
  --image_size 512 \
  --resize_mode no
```

After downloading the dataset, it is expected to be the following format:

```text
cc12m/
|-- 00000.tar
|-- 00001.tar
`-- ...
```

Where each `.tar` contains paired files:

```text
000000001.jpg
000000001.txt
000000002.jpg
000000002.txt
```

During training, point `CC12M_ROOT` in `settings.py` at the shard pattern:

```python
CC12M_ROOT = "/path/to/cc12m/{00000..01096}.tar"
# or
CC12M_ROOT = "gs://your-bucket/data/cc12m/{00000..01096}.tar" # we support paths on GCS bucket
```

## 120K Mix Fine-Tuning

MiniT2I fine-tunes on a public high-quality mixture:

- [BLIP3o-60K](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k)
- [DALL-E 3 samples](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset)
- [ShareGPT-4o-Image](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image)

Same as before, all three sources must be prepared as flat WebDataset tar files with `{key}.jpg` and `{key}.txt` pairs.

### BLIP3o-60K

BLIP3o-60K is already released as WebDataset shards, so download it directly:

```bash
python scripts/datasets/download_blip3o_60k.py \
  --output-dir /path/to/datasets/BLIP3o-60k
```

To upload while downloading:

```bash
python scripts/datasets/download_blip3o_60k.py \
  --output-dir /path/to/staging/BLIP3o-60k \
  --gcs-bucket gs://your-bucket/data/BLIP3o-60k \
  --delete-after-upload
```

### DALL-E 3

DALL-E 3 samples are released as parquet files and is converted into `shard-XXX.tar`:

```bash
python scripts/datasets/prepare_dalle3_webdataset.py \
  --raw-dir /path/to/raw/dalle3 \
  --output-dir /path/to/datasets/dalle3
```

### ShareGPT-4o-Image

ShareGPT-4o-Image is released as metadata plus image tar parts. The script filters prompts longer than 256 FLAN-T5 tokens, converts PNG images to JPEG, and writes `shard-XXX.tar`:

```bash
python scripts/datasets/prepare_sharegpt4o_webdataset.py \
  --meta-dir /path/to/raw/gpt4oimg \
  --output-dir /path/to/datasets/gpt4oimg \
  --tokenizer google/flan-t5-large \
  --max-tokens 256
```

### During Fine-Tuning

Set the fine-tuning roots in `settings.py`:

```python
BLIP3_FT60K_ROOT = "/path/to/datasets/BLIP3o-60k"
DALLE3_ROOT = "/path/to/datasets/dalle3/shard-{000..013}.tar"
SHAREGPT4O_ROOT = "/path/to/datasets/gpt4oimg/shard-{000..013}.tar"
```

For BLIP3o-60K, our code expands the dataset root to the fixed tar names listed in [`utils/data_util.py`](utils/data_util.py), for example `dalle3.tar`, `geneval_train.tar`, `human_gestures.tar`, `journeyDB.tar`, `mscoco_human.tar`, `object_*.tar`, `occupation_*.tar`, and `text_*.tar`. `use_dalle3` and `use_sharegpt4o` read their configured tar shard patterns directly.
