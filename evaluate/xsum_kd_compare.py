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

# label -> (model size, adapter directory name, or "" for the untrained model)
MODELS = {
    "untrained 4B": ("4b", ""),
    "reference LoRA": ("4b", "gemma3-xsum_screen-full-lora"),
    "self-distilled 4-bit": ("4b", "gemma3-xsum_kd_screen-lora"),
    "self-distilled 8-bit": ("4b", "gemma3-xsum_kd_screen_8bit-lora"),
    "self-distilled bf16 (own loop)": ("4b", "gemma3-xsum_kd_screen_bf16loop-lora"),
    "self-distilled bf16, BBC teacher": ("4b", "gemma3-xsum_kd_screen_bbc_bf16loop-lora"),
    "untrained 1B": ("1b", ""),
    "reference LoRA 1B": ("1b", "gemma3-xsum_screen-full_1b-lora"),
    "distilled 4B->1B (own loop)": ("1b", "gemma3-xsum_kd_screen_1b_bf16loop-lora"),
    "distilled 4B->1B, BBC teacher": ("1b", "gemma3-xsum_kd_screen_bbc_1b_bf16loop-lora"),
    # Gold targets: CE on the reference, KD on the BBC-prompted teacher scoring
    # it; CE weight swept, logit adjustment (LA) on and off, CE 1.00 = no KD.
    "gold, CE 0.50": ("4b", "gemma3-xsum_kd_gold_ce50-lora"),
    "gold, CE 0.50 + LA": ("4b", "gemma3-xsum_kd_gold_ce50_la-lora"),
    "gold, CE 0.75": ("4b", "gemma3-xsum_kd_gold_ce75-lora"),
    "gold, CE 0.75 + LA": ("4b", "gemma3-xsum_kd_gold_ce75_la-lora"),
    "gold, CE 0.90": ("4b", "gemma3-xsum_kd_gold_ce90-lora"),
    "gold, CE 0.90 + LA": ("4b", "gemma3-xsum_kd_gold_ce90_la-lora"),
    "gold, CE 1.00": ("4b", "gemma3-xsum_kd_gold_ce100-lora"),
    # The 27B (a local snapshot directory), untrained and trained on the same
    # 4,000 references with QLoRA - the strong XSum teacher candidate.
    "untrained 27B": ("27b", ""),
    "reference QLoRA 27B": ("27b", "gemma3-xsum_screen-full_27b-lora"),
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
        if log.status != "success" or log.eval.task.split("/")[-1] != "xsum":
            continue
        size = ("1b" if "gemma-3-1b" in log.eval.model else "4b" if "gemma-3-4b" in log.eval.model
                else "27b" if "gemma-3-27b" in log.eval.model else None)
        if "xsum_screen" not in str((log.eval.task_args or {}).get("dataset_path", "")):
            continue
        adapter = str((log.eval.model_args or {}).get("adapter_path", "")).rstrip("/")
        for name, (want_size, stem) in MODELS.items():
            if size != want_size:
                continue
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

    pairs = [("reference QLoRA 27B", "reference LoRA"), ("untrained 27B", "untrained 4B"),
             *[(f"gold, CE {w} + LA", f"gold, CE {w}") for w in ("0.50", "0.75", "0.90")],
             *[(f"gold, CE {w}", "gold, CE 1.00") for w in ("0.50", "0.75", "0.90")],
             ("gold, CE 1.00", "reference LoRA"),
             ("distilled 4B->1B, BBC teacher", "distilled 4B->1B (own loop)"),
             ("distilled 4B->1B, BBC teacher", "reference LoRA 1B"),
             ("self-distilled bf16, BBC teacher", "self-distilled bf16 (own loop)"),
             ("self-distilled bf16, BBC teacher", "reference LoRA"),
             ("self-distilled bf16 (own loop)", "self-distilled 4-bit"),
             ("self-distilled bf16 (own loop)", "reference LoRA"),
             ("self-distilled 8-bit", "self-distilled 4-bit"),
             ("distilled 4B->1B (own loop)", "reference LoRA 1B"),
             ("distilled 4B->1B (own loop)", "self-distilled bf16 (own loop)")]
    for a, b in pairs:
        if a in picked and b in picked:
            for m in ("rouge1", "semantic", "entity_support"):
                g, lo, hi, n = paired_bootstrap(values(picked[b], m), values(picked[a], m))
                print(f"{a} minus {b}, {m}: {g:+.4f} [{lo:+.4f}, {hi:+.4f}] n={n}")


if __name__ == "__main__":
    main()
