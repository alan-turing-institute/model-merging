"""Expose axolotl's v1 (legacy) KD strategy under a `load` entry point.

WHY THIS EXISTS. axolotl 0.18.0 ships two KD strategies in
`axolotl.integrations.kd.chat_template`: `load` returns the v2 strategy, which
expects a dataset that already carries `target_token_ids`/`target_mask`, and
`load_legacy` returns v1, which builds those from per-position top-k logprobs -
our format, and the format of axolotl's own published KD dataset.

There is no `type:` string that reaches `load_legacy`. The resolver in
`axolotl.prompt_strategies.load` does strip a trailing `load_*` and use it as
the function name, but taking that branch skips the branch that corrects the
package, so `package` stays "axolotl.prompt_strategies" and it tries to import
`axolotl.prompt_strategies.axolotl.integrations.kd.chat_template`. That raises
ModuleNotFoundError, the resolver returns None, and the run dies with
"unhandled prompt tokenization strategy". The suffix convention only works for
strategies that live inside axolotl.prompt_strategies.

A plain module does resolve, because a dotted path without a `load_*` suffix
takes the branch that imports the parent package properly. So this exposes
`load` and delegates. Referenced as `type: kd_strategies.legacy`; axolotl runs
with train/ as the working directory, and `python -m` puts that on sys.path.
"""

import inspect

from axolotl.integrations.kd.chat_template import load_legacy


def load(tokenizer, cfg, ds_cfg=None, processor=None):
    # Forward only what the underlying loader actually accepts - its signature
    # has changed across versions, and passing an unexpected keyword here would
    # fail in the same opaque way the problem this works around does.
    signature = inspect.signature(load_legacy)
    kwargs = {}
    if "ds_cfg" in signature.parameters:
        kwargs["ds_cfg"] = ds_cfg
    if "processor" in signature.parameters:
        kwargs["processor"] = processor
    return load_legacy(tokenizer, cfg, **kwargs)
