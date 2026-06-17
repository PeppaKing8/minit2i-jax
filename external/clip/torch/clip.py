import hashlib
import os
import urllib
import warnings
from typing import List, Union

import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}


class _StateDictModel:
    def __init__(self, state_dict: dict, input_resolution: int):
        self._state_dict = state_dict
        self.visual = type("VisualInfo", (), {"input_resolution": input_resolution})()

    def state_dict(self):
        return self._state_dict


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        total = int(source.info().get("Content-Length"))
        with tqdm(total=total, ncols=80, unit="iB", unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Downloaded CLIP checkpoint failed SHA256 verification")
    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    return list(_MODELS.keys())


def load(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit: bool = False,
    download_root: str = None,
):
    if jit:
        warnings.warn("The vendored CLIP loader ignores jit=True and loads a state_dict model.")
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, "rb") as opened_file:
        try:
            state_dict = torch.jit.load(opened_file, map_location="cpu").state_dict()
        except RuntimeError:
            opened_file.seek(0)
            state_dict = torch.load(opened_file, map_location="cpu")
    if "visual.proj" not in state_dict:
        raise ValueError("Only ViT CLIP checkpoints are supported by this vendored loader.")
    patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    input_resolution = patch_size * grid_size
    model = _StateDictModel(state_dict, input_resolution)
    return model, _transform(model.visual.input_resolution)


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if not truncate:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            tokens = tokens[:context_length]
            tokens[-1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
