"""An Inspect AI model provider for a local HF model plus a PEFT/LoRA adapter.

Inspect's built-in `hf` provider loads weights with `AutoModelForCausalLM` and
has no notion of adapters - there is not one mention of peft, lora or adapter
anywhere in it. The crime models are base+adapter, so without this every
adapter would have to be materialised into a full ~8.1GB model purely to be
scored, which is most of a compute instance's disk per variant. (The XSum
models are self-distilled full models and use the plain hf provider.)

Registered as `hf-peft`, so a model reference looks like:

    hf-peft/google/gemma-3-4b-it        with model arg adapter_path=<path>

Inspect finds it through the `inspect_ai` entry point in pyproject.toml, so no
import is needed before `inspect eval`.

Everything else - `batch_size`, `device`, `do_sample`, `model_path`, chat
template handling - is inherited from the built-in provider unchanged, so this
stays a thin shim rather than a fork.

NOTE: this subclasses `inspect_ai.model._providers.hf.HuggingFaceAPI`, which is
a private module. That is deliberate - reimplementing the provider to add four
lines would be worse - but it means an inspect-ai upgrade can break it. The
import is guarded so the failure is a clear message rather than an
AttributeError deep inside a generate call.
"""

from pathlib import Path
from typing import Any

from inspect_ai.model import GenerateConfig, modelapi

try:
    from inspect_ai.model._providers.hf import HuggingFaceAPI
except ImportError as exc:  # pragma: no cover - only on an incompatible upgrade
    raise ImportError(
        "Could not import HuggingFaceAPI from inspect_ai.model._providers.hf. "
        "This is a private module and may have moved in a newer inspect-ai; "
        "hf_peft_provider.py needs updating to match."
    ) from exc


def adapter_vocab_size(adapter_path) -> int | None:
    """Rows of the embedding matrix saved in a local adapter, if it saved one.

    Axolotl resizes the embeddings to len(tokenizer) when the tokenizer is the
    larger of the two, and PEFT then saves the whole embedding and output layers
    with the adapter. Gemma 3 1B is such a model - 262,145 tokens against
    262,144 embedding rows - while Gemma 3 4B's embedding is larger than its
    tokenizer and its adapters save no embeddings at all.
    """
    file = Path(adapter_path) / "adapter_model.safetensors"
    if not file.is_file():
        return None
    from safetensors import safe_open

    with safe_open(str(file), "pt") as f:
        for key in f.keys():
            if key.endswith("embed_tokens.weight"):
                return f.get_slice(key).get_shape()[0]
    return None


class HuggingFacePeftAPI(HuggingFaceAPI):
    """The built-in HF provider, with a PEFT adapter applied after loading."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ):
        # Pulled out before the parent sees it - the parent forwards unknown
        # model_args straight to from_pretrained, which would reject it.
        adapter_path = model_args.pop("adapter_path", None)

        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
            **model_args,
        )

        if adapter_path:
            from peft import PeftModel

            # Resize the base model to the adapter's vocabulary first, exactly
            # as training did, or loading fails with a size mismatch on
            # embed_tokens and lm_head. A no-op for adapters without saved
            # embeddings.
            vocab = adapter_vocab_size(adapter_path)
            if vocab and vocab != self.model.get_input_embeddings().num_embeddings:
                self.model.resize_token_embeddings(vocab)

            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
            self.model.eval()


@modelapi(name="hf-peft")
def hf_peft() -> type[HuggingFacePeftAPI]:
    return HuggingFacePeftAPI
