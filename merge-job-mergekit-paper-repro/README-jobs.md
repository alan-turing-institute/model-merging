
## Passing the HuggingFace token (2026-09-19)

`epfl-llm/meditron-7b` is gated, so every job that downloads it needs a real
token. Eight jobs here plus the MoE merge reference `${{inputs.hf_token}}` in
their command.

They used to declare it in the YAML:

```yaml
inputs:
  hf_token:
    type: string
    default: ""
```

That is *component* syntax, not *job* syntax — a job input takes a value, not a
type and a default. An older `ml` extension tolerated it; 2.44.1 does not, and
rejects the file before submission. Supplying an empty value instead fails at
the service with "The Value field is required".

So the declarations are gone and the token is supplied at submit time. Keep it
in an environment variable rather than typing it into a command line, so it
stays out of shell history and out of the YAML:

```bash
export HF_TOKEN=...            # once per shell, from your HF account settings
az ml job create -f job-merge-single-moe.yml -g tire-1 -w tire-2 \
  --set inputs.hf_token=$HF_TOKEN
```

Jobs touching only `NousResearch/Llama-2-7b-chat-hf` (an ungated mirror) do not
need it and carry no `${{inputs.hf_token}}` reference.
