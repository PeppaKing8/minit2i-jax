#!/bin/bash
set -euo pipefail

# TPU-oriented install. Adjust the JAX version to match your TPU runtime if
# your environment already provides a newer supported JAX stack.
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements.txt
