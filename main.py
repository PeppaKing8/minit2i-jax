import jax

from absl import app, flags
from ml_collections import config_flags

import warnings

warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_string(
    "mode", None, "Run mode (e.g. local_debug, remote_run); copied to config.mode."
)
flags.DEFINE_string(
    "load_from",
    "",
    "Optional checkpoint path overriding config.load_from for resume/evaluation.",
)
flags.DEFINE_string(
    "load_pt_from",
    "",
    "Optional teacher checkpoint path for Mean Flow distillation.",
)

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def _install_jax_tree_shims():
    """Install legacy tree aliases after distributed initialization."""
    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree_util.tree_map
    if not hasattr(jax, "tree_leaves"):
        jax.tree_leaves = jax.tree_util.tree_leaves
    if not hasattr(jax, "tree_flatten"):
        jax.tree_flatten = jax.tree_util.tree_flatten
    if not hasattr(jax, "tree_unflatten"):
        jax.tree_unflatten = jax.tree_util.tree_unflatten


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    jax.distributed.initialize()
    _install_jax_tree_shims()

    print("Starting MiniT2I run.", flush=True)

    from utils import logging_util
    from utils.logging_util import log_for_0

    logging_util.supress_checkpt_info()

    if not getattr(FLAGS.config, "mean_flow_distill", False):
        raise ValueError(
            "The mean_flow_distill branch only supports Mean Flow distillation configs. "
            "Use the main branch for diffusion pretraining/fine-tuning."
        )
    import train_mean_flow as train

    log_for_0("JAX process: %d / %d", jax.process_index(), jax.process_count())
    log_for_0("JAX local devices: %r", jax.local_devices())
    log_for_0("FLAGS.config: \n{}".format(FLAGS.config))

    FLAGS.config.mode = FLAGS.mode
    if FLAGS.load_from:
        FLAGS.config.load_from = FLAGS.load_from
    if FLAGS.load_pt_from:
        FLAGS.config.load_pt_from = FLAGS.load_pt_from

    train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
