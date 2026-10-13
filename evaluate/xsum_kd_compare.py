"""Compare the XSum models on the xsum_screen test split.

    uv run python xsum_kd_compare.py logs

Untrained gemma-3-4b-it, the reference-trained LoRA and each self-distilled
LoRA, scored on the same 1,000 articles: ROUGE, semantic similarity, entity
support (overall, and on outputs that do not loop), length, and the share of
outputs repeating one word three or more times in a row. Then paired bootstrap
differences between the self-distilled variants and against the reference LoRA.
"""

import re
import statistics
import sys

from inspect_ai.log import list_eval_logs, read_eval_log

from scaling_curve import paired_bootstrap

MODELS = {
    "untrained 4B": "",
    "reference LoRA": "gemma3-xsum_screen-full-lora",
    "self-distilled 4-bit": "gemma3-xsum_kd_screen-lora",
    "self-distilled 8-bit": "gemma3-xsum_kd_screen_8bit-lora",
    "self-distilled bf16 (own loop)": "gemma3-xsum_kd_screen_bf16loop-lora",
}
REPEAT = re.compile(r"\b(\w+)\b(?:\W+\1\b){2,}", re.IGNORECASE)


def values(log, metric):
    out = {}
    for s in log.samples:
        for sc in (s.scores or {}).values():
            v = sc.value.get(metric) if isinstance(sc.value, dict) else None
            if isinstance(v, (int, float)) and v == v:
                out[s.id] = float(v)
    return out


def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "logs"
    picked = {}
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info)
        if log.status != "success" or log.eval.task.split("/")[-1] != "xsum" or "gemma-3-4b" not in log.eval.model:
            continue
        if "xsum_screen" not in str((log.eval.task_args or {}).get("dataset_path", "")):
            continue
        adapter = str((log.eval.model_args or {}).get("adapter_path", "")).rstrip("/")
        for name, stem in MODELS.items():
            if (stem and adapter.endswith(stem)) or (not stem and not adapter):
                if name not in picked or log.eval.created > picked[name].eval.created:
                    picked[name] = log

    print(f"{'model':32} {'R1':>7} {'R2':>7} {'RL':>7} {'sem':>7} {'entity':>7} {'ent.clean':>9} {'words':>6} {'repeat':>7}")
    for name in MODELS:
        log = picked.get(name)
        if not log:
            print(f"{name:32} (no log yet)")
            continue
        mean = lambda m: statistics.fmean(values(log, m).values())
        looping = {s.id for s in log.samples if REPEAT.search(s.output.completion or "")}
        clean = [v for i, v in values(log, "entity_support").items() if i not in looping]
        print(f"{name:32} {mean('rouge1'):7.4f} {mean('rouge2'):7.4f} {mean('rougeL'):7.4f} {mean('semantic'):7.4f} "
              f"{mean('entity_support'):7.4f} {statistics.fmean(clean):9.4f} {mean('pred_words'):6.1f} "
              f"{len(looping) / len(log.samples):7.1%}")

    pairs = [("self-distilled 8-bit", "self-distilled 4-bit"), ("self-distilled 8-bit", "reference LoRA"),
             ("self-distilled bf16 (own loop)", "self-distilled 8-bit")]
    for a, b in pairs:
        if a in picked and b in picked:
            for m in ("rouge1", "semantic", "entity_support"):
                g, lo, hi, n = paired_bootstrap(values(picked[b], m), values(picked[a], m))
                print(f"{a} minus {b}, {m}: {g:+.4f} [{lo:+.4f}, {hi:+.4f}] n={n}")


if __name__ == "__main__":
    main()
