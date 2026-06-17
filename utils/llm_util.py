import jax
import numpy as np
import gc
from transformers import AutoTokenizer
from models.t5_encoder import T5Config
from models.t5_encoder import create_t5_encode_fn
from utils.logging_util import log_for_0

class LLM:
    
    def __init__(self, config):
        self.config = config
        self.model_name = config.dataset.llm
        assert self.model_name in [
            'google/flan-t5-small',
            'google/flan-t5-base', 
            'google/flan-t5-large',
            'google/flan-t5-xxl',
            'debug-llm'
        ], f'Unsupported model: {self.model_name}'

        self.tokenizer = None  # lazily initialized to avoid fork-after-init warnings

        if self.model_name == 'debug-llm':
            self.model_config = T5Config(d_model=16, d_kv=16, d_ff=16, num_layers=1, num_heads=1)
        else:
            self.model_config = T5Config.from_pretrained(self.model_name)

        self.hidden_dim = self.model_config.d_model

        # encoder are initialized later
        self.encode_fn = None
        self.model = None
        self.params = None

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name if self.model_name != 'debug-llm' else 'google/flan-t5-base'
            )
    
    def init_encoder(self, mesh_bundle):
        """
        Initialize encoder after DataLoader is constructed to avoid forking a large model.
        """
        if self.encode_fn is not None:
            return  # already initialized

        log_for_0(f'Building LLM encoder {self.model_name} after dataloader setup...')
        log_for_0(f"Before loading LLM encoder, memory allocated: {jax.device_get(jax.local_devices()[0].memory_stats()['bytes_in_use']) / (1024**3):.2f} GB")

        self.encode_fn, self.model, self.params = create_t5_encode_fn(
            model_name=self.model_name,
            max_encoder_length=self.config.dataset.prompt_length,
            mesh_bundle=mesh_bundle,
            model_config=self.model_config,
        )

        # remove decoder to save memory
        if isinstance(self.params, dict) and "params" in self.params and "decoder" in self.params["params"]:
            del self.params["params"]["decoder"]
        if isinstance(self.params, dict) and "decoder" in self.params:
            del self.params["decoder"]
        if hasattr(self.model, "decoder"):
            delattr(self.model, "decoder")
        gc.collect()
        
        log_for_0(f'After loading LLM encoder, memory allocated: {jax.device_get(jax.local_devices()[0].memory_stats()["bytes_in_use"]) / (1024**3):.2f} GB')
        
    def tokenize_single(self, text):
        self._ensure_tokenizer()
        o = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.dataset.prompt_length,
        )
        input_ids, attention_mask = o.input_ids, o.attention_mask
        # input_ids may have shape (1, seq_length), we need to squeeze it
        if input_ids.ndim != 1:
            assert input_ids.ndim == 2 and input_ids.shape[0] == 1, f'Unexpected input_ids shape: {input_ids.shape}'
            input_ids = input_ids.squeeze(0)
        if attention_mask.ndim != 1:
            assert attention_mask.ndim == 2 and attention_mask.shape[0] == 1, f'Unexpected attention_mask shape: {attention_mask.shape}'
            attention_mask = attention_mask.squeeze(0)
        return input_ids, attention_mask
    
    def tokenize_batch(self, texts, to_np=True):
        self._ensure_tokenizer()
        input_ids, attention_masks = zip(*[self.tokenize_single(text) for text in texts])
        if to_np:
            input_ids = np.stack([x.numpy() for x in input_ids], axis=0)
            attention_masks = np.stack([x.numpy() for x in attention_masks], axis=0)
        else:
            raise NotImplementedError("Only NumPy token batches are supported.")
        return input_ids, attention_masks
