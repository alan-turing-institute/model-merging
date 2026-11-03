# Exposes Axolotl's own legacy KD loader under a name that `type:` can reach.
#
# The example dataset stores logprobs in the legacy format, which only
# `axolotl.integrations.kd.chat_template.load_legacy` understands, but the
# prompt-strategy resolver only honours a `.load_*` suffix for modules inside
# axolotl.prompt_strategies. The plugin README's `type:` therefore reaches the
# v2 loader and fails with KeyError: 'target_token_ids'. Needs PYTHONPATH=.
from axolotl.integrations.kd.chat_template import load_legacy as load  # noqa: F401
