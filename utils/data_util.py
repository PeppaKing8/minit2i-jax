from .logging_util import log_for_0


_BLIP3_FT60K_TARS = [
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
]


def resolve_dataset_roots(cfg):
    roots = []

    def _maybe_add(use_key, root_key, label):
        if not getattr(cfg.dataset, use_key, False):
            return
        root = getattr(cfg.dataset, root_key)
        assert root, f"{label} dataset is enabled but cfg.dataset.{root_key} is empty."
        roots.append(root)

    _maybe_add('use_cc12m', 'cc12m_root', 'CC12M')
    _maybe_add('use_laion_aes', 'laion_aes_root', 'LAION-Aesthetic')
    _maybe_add('use_blip3_short', 'blip3_short_root', 'BLIP-3o Short-Caption')
    _maybe_add('use_blip3_long', 'blip3_long_root', 'BLIP-3o Long-Caption')
    _maybe_add('use_blip3_journey', 'blip3_journey_root', 'BLIP-3o JourneyDB')

    # BLIP-3o 60k fine-tuning fans out to a fixed list of .tar files instead of
    # a single root, so it doesn't fit the simple template.
    if getattr(cfg.dataset, "use_blip3_ft60k", False):
        base = cfg.dataset.blip3_ft60k_root
        assert base, "BLIP-3o 60k fine-tuning dataset is enabled but cfg.dataset.blip3_ft60k_root is empty."
        roots.append([f"{base}/{f}" for f in _BLIP3_FT60K_TARS])

    _maybe_add('use_dalle3', 'dalle3_root', 'DALL-E 3')
    _maybe_add('use_sharegpt4o', 'sharegpt4o_root', 'ShareGPT4o')

    user_weights = getattr(cfg.dataset, "mix_weights", [])
    if user_weights:
        assert len(user_weights) == len(roots), (
            f"Number of user-specified mix weights {len(user_weights)} "
            f"does not match number of dataset roots {len(roots)}"
        )
        weights = user_weights
    else:
        log_for_0("!!! WARNING: No or invalid user-specified dataset mix weights, using uniform weights.")
        weights = [1.0] * len(roots)
    return roots, weights
