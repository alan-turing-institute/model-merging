# An Axolotl plugin that seeds every RNG before the model is built, so a run's
# `seed` also fixes the LoRA adapter's initial weights.
#
# Axolotl passes `seed` only to the Trainer and its dataset shuffles. The Trainer
# seeds itself when it is created, after the model and adapter are loaded, so
# without this the adapter's A matrices are drawn from an unseeded RNG and two
# runs with the same seed start from different weights. pre_model_load runs
# before the model and adapter are created (axolotl/loaders/model.py).
#
# train.sbatch and train_kd.sbatch add it to the run config's `plugins` along
# with the seed, and put the repo root on PYTHONPATH so Axolotl can import it.
# Harmless for full fine-tunes (XSum), which have no adapter to initialise.
from axolotl.integrations.base import BasePlugin
from transformers import set_seed


class SeedPlugin(BasePlugin):
    def pre_model_load(self, cfg):
        if cfg.seed is not None:
            set_seed(cfg.seed)
            print(f"seed_plugin: set_seed({cfg.seed}) before loading the model")
