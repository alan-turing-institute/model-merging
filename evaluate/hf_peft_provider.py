"""An Inspect AI model provider for a local HF model plus a PEFT/LoRA adapter.

Inspect's built-in `hf` provider loads weights with `AutoModelForCausalLM` and
has no notion of adapters - there is not one mention of peft, lora or adapter
anywhere in it. Most of what this repo evaluates is base+adapter, so without
this every adapter would have to be materialised into a full ~8.1GB model
purely to be scored, which is most of a compute instance's disk per variant.

Registered as `hf-peft`, so a model reference looks like:

    hf-peft/google/gemma-3-4b-it        with model arg adapter_path=<path>

Everything else - `batch_size`, `device`, `do_sample`, `model_path`, chat
template handling - is inherited from the built-in provider unchanged, so this
stays a thin shim rather than a fork.

NOTE: this subclasses `inspect_ai.model._providers.hf.HuggingFaceAPI`, which is
a private module. That is deliberate - reimplementing the provider to add four
lines would be worse - but it means an inspect-ai upgrade can break it. The
import is guarded so the failure is a clear message rather than an
AttributeError deep inside a generate call.
"""

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

            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
            self.model.eval()


@modelapi(name="hf-peft")
def hf_peft() -> type[HuggingFacePeftAPI]:
    return HuggingFacePeftAPI
