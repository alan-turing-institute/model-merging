# Writes the routing study of the Mistral task MoE as a markdown report, from
# results/task_merging_routing/mistral/ (export_moe_routing.py: the rebuilt MoE
# with main's router and three new ones, its experts and the base model, seeds
# 1-2, and the routing probes of merge/moe_routing.py) and, for comparison,
# main's own MoE in results/task_merging/mistral/.
#
# Usage, from evaluate/:
#   python make_moe_routing_report.py ../results/moe-routing-mistral.md
import json
import statistics as st
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
OURS, MAIN = RES / "task_merging_routing" / "mistral", RES / "task_merging" / "mistral"
SEEDS = [1, 2]
TASKS = {"crime": ("accuracy/mean", "Crime accuracy", 3), "wine": ("abs_error/mean", "Wine MAE", 3),
         "clinc150": ("accuracy/mean", "CLINC150 accuracy", 3), "xsum": ("rouge1/mean", "XSum ROUGE-1", 3)}
ROUTERS = {"moe": "main's router (mergekit hidden, main's prompts)",
           "moe_prompts": "1. real chat-formatted examples as mergekit's prompts",
           "moe_fit": "2. fitted on hidden states (logistic regression)",
           "moe_train": "3. trained (routing + answer loss)"}
EXPERTS = ["crime", "wine", "CLINC150", "XSum"]


def metric(path, task):
    return json.loads(path.read_text())["metrics"][TASKS[task][0]] if path.exists() else None


def fmt(x, k=3):
    return "–" if x is None else f"{x:.{k}f}"


def recovery(seed, router):
    """R per task, (merged - base) / (expert of the seed - base), and their mean."""
    rs = []
    for t in TASKS:
        m, e, b = (metric(OURS / f"seed_{seed}/{router}/{t}.json", t), metric(OURS / f"seed_{seed}/expert_{t}.json", t),
                   metric(OURS / f"base_{t}.json", t))
        rs.append(None if None in (m, e, b) else (m - b) / (e - b))
    return rs + [st.mean(rs) if None not in rs else None]


def main(out):
    lines = ["# Routing the task MoE: Mistral 7B, seeds 1-2", "",
             "Main's sparse task MoE (`merge/moe/mistral_task_moe_4_seed_<S>_config.yaml`: four MLP-only experts, "
             "crime, wine, CLINC150 with the labels prompt and XSum by self-distillation, one expert per token, "
             "mergekit's `gate_mode: hidden`) keeps most of its experts' gains on wine and CLINC150 but loses much "
             "of crime's. This study rebuilt it and asked whether a better router recovers crime. The experts and "
             "the MoE were rebuilt from main (7122af6) with its configs, seeds 1 and 2, because main's models are "
             "not readable to us; every router below routes the same four experts of a seed, and is fitted on "
             "training data only. Each MoE is evaluated as main evaluates its MoEs (test sets in full; CLINC150 "
             "with the labels prompt, XSum with the long one).", "",
             "*R* is the share of the expert's gain over the base model a model keeps, for one task and seed: "
             "(model - base) / (expert of the same seed - base), on the task's main metric (wine's MAE is lower-"
             "better, which the ratio handles). 1 is as good as the expert, 0 no better than the base model. "
             "*All four* is the mean of a seed's four R values.", "",
             "## The rebuild", "",
             "Main's MoE (its own experts) and the rebuild (ours), same config and code:", "",
             "| | seed | " + " | ".join(v[1] for v in TASKS.values()) + " |", "|---" * (len(TASKS) + 2) + "|"]
    for name, root in (("main's MoE", MAIN), ("rebuilt MoE", OURS)):
        for s in SEEDS:
            lines.append(f"| {name} | {s} | " + " | ".join(fmt(metric(root / f"seed_{s}/moe/{t}.json", t))
                                                          for t in TASKS) + " |")
    lines += ["", "The rebuild's crime accuracy sits inside main's five seeds (0.835-0.941), so it reproduces the "
              "weakness rather than an accident of training.", "",
              "## Where crime's routing fails", "",
              "The probe (`merge/moe_routing.py probe`, 300 test prompts per task) records the expert each layer "
              "picks for every prompt token. A crime answer is decided at the answer position, the last prompt "
              "token. Share of routing decisions there going to each expert, over all 32 layers, and the share "
              "going home in layers 0-7; for crime also split by whether the MoE answered correctly:", "",
              "| router | seed | " + " | ".join(f"to {e}" for e in EXPERTS)
              + " | home, layers 0-7 | home when right (n) | home when wrong (n) |", "|---" * 9 + "|"]
    for r in ROUTERS:
        for s in SEEDS:
            p = OURS / f"seed_{s}/{r}/probe.json"
            if not p.exists():
                continue
            c = json.loads(p.read_text())["crime"]
            share = [st.mean(row[e] for row in c["answer_expert_share"]) for e in range(4)]
            ok, bad = c["correct"], c["wrong"]
            lines.append(f"| {r} | {s} | " + " | ".join(f"{x:.2f}" for x in share)
                         + f" | {st.mean(c['answer_home_by_layer'][:8]):.2f}"
                         + f" | {st.mean(ok['answer_home_by_layer']):.2f} ({ok['n']}) | "
                         + f"{st.mean(bad['answer_home_by_layer']):.2f} ({bad['n']}) |")
    lines += ["", "With main's router, about half of the decisions at crime's answer position go to the wine and "
              "CLINC150 experts, nearly all of them in the early layers, and the wrong answers are routed home "
              "less often than the right ones. mergekit's hidden router is built from the base model's state at "
              "the last token of each positive prompt, and main's prompts are bare instructions and post titles "
              "without the chat template, so no expert's router row stands for the answer position. Main found "
              "the same independently (fb26f01).", "",
              "## Routers", "",
              "1. **Better prompts** (`moe_routing.py prompts`): mergekit as before, but each expert's positive "
              "prompts are 32 real training examples rendered with the chat template, 16 ending at the answer "
              "position and 16 cut at a random token (`merge/moe/mistral_task_moe_4_seed_<S>_prompts_config.yaml`).",
              "2. **Fitted** (`moe_routing.py fit`): each layer's router is a multinomial logistic regression, "
              "without bias like the router, of the task on the base model's router input (post-attention, "
              "normalised hidden state), over 300 training sequences per task (prompt and answer), 20,000 tokens "
              "per task and layer, the answer position always included.",
              "3. **Trained** (`moe_routing.py train`): the routers alone (32 x 4 x 4096 = 524,288 parameters; "
              "everything else frozen) trained on the MoE itself for 600 steps, tasks in turn, from main's router: "
              "a per-token routing loss towards the sequence's task, plus the answer tokens' loss. With one expert "
              "per token the chosen expert's weight is renormalised to 1, so the answer loss reaches the router "
              "only through a straight-through weight (top / top.detach(), 1 in the forward pass).",
              "4. **Per prompt**: route every token of a prompt to one expert, chosen once per prompt. An MoE that "
              "sends every token to expert e is expert e's model, since the experts differ from the base only in "
              "their MLPs (main checked this for its MoEs: argmax agreement 0.97-1.00), so this router scores as "
              "the expert on every prompt it assigns correctly; what remains to measure is that assignment (below).",
              "", "## Results", "",
              "| model | seed | " + " | ".join(v[1] for v in TASKS.values()) + " | R crime | R wine | R CLINC150 "
              "| R XSum | All four |", "|---" * (len(TASKS) + 7) + "|"]
    lines.append("| base model | – | " + " | ".join(fmt(metric(OURS / f"base_{t}.json", t)) for t in TASKS)
                 + " | | | | | |")
    for s in SEEDS:
        lines.append(f"| experts | {s} | " + " | ".join(fmt(metric(OURS / f"seed_{s}/expert_{t}.json", t))
                                                       for t in TASKS) + " | 1 | 1 | 1 | 1 | 1 |")
    for r, label in ROUTERS.items():
        for s in SEEDS:
            lines.append(f"| {label} | {s} | " + " | ".join(fmt(metric(OURS / f"seed_{s}/{r}/{t}.json", t))
                                                          for t in TASKS)
                         + " | " + " | ".join(fmt(x, 2) for x in recovery(s, r)) + " |")
    lines += ["", "## Routing per prompt", "",
              "Share of each task's test prompts whose mean router probability (over tokens and layers) picks "
              "the task's own expert, i.e. how often routing per prompt would use the right expert:", "",
              "| router | seed | " + " | ".join(EXPERTS) + " |", "|---" * 6 + "|"]
    for r in ROUTERS:
        for s in SEEDS:
            p = OURS / f"seed_{s}/{r}/probe.json"
            if p.exists():
                d = json.loads(p.read_text())
                lines.append(f"| {r} | {s} | " + " | ".join(f"{d[t]['prompt_choice_home']:.2f}" for t in TASKS) + " |")
    lines += ["", "With any of the three new routers every probed prompt is assigned to its own expert, so routing "
              "per prompt would score as the experts do (R = 1). With main's router it would misroute 20-42% of "
              "crime prompts.", "",
              "## Caveats", "",
              "- Two seeds and one model (Mistral 7B); main's MoE has five seeds and also Llama 3.1 8B.",
              "- The probe uses 300 test prompts per task; the evaluations use the full test sets.",
              "- XSum for router 1, seed 2 ran at batch size 1 (the MoE ran out of GPU memory at 2); decoding is "
              "greedy, so only the padding differs.",
              "- Routers 2 and 3 were fitted and trained on the training splits only; router 3's progress was "
              "checked on 40 validation sequences per task, never on test data.", "",
              "## Reproducing", "",
              "On Isambard, in a worktree of main: `merge/submit_moe_rebuild.sh` (experts, conversions, base and "
              "expert evaluations, main's `submit_moe.sh`), then `merge/submit_routing.sh` (probes, the three "
              "routers and their evaluations); `evaluate/export_moe_routing.py` writes the JSONs and this script "
              "the report."]
    Path(out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
