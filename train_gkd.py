"""On-policy distillation (GKD) — the online arm.

Axolotl cannot do this. Its KD trainer requires precomputed teacher logprobs in
every batch, at every version including main; the kd_online_* config fields are
placeholders the trainer never reads. Its `rl:` key offers dpo/ipo/kto/simpo/
orpo/grpo/ebft and no distillation. So the online arm is driven through TRL's
GKDTrainer instead.

WHAT MAKES THIS DIFFERENT FROM THE OFFLINE ARM, and why it is worth a separate
run rather than being the same thing delivered live: offline distillation
scores the DATASET's gold sequences. GKD scores the STUDENT's own generations.
That is the train/inference distribution mismatch the method exists to fix -
with a frozen teacher, a merely "live-served" teacher would return identical
targets to our precomputed ones and the arm would compare infrastructure, not
methods.

  --lmbda   fraction of on-policy (student-generated) data. 1.0 = fully
            on-policy, 0.0 = supervised JSD on the dataset's sequences.
  --beta    generalized JSD interpolation. 0.0 -> forward KL, 1.0 -> reverse KL.

GEMMA WARNING: TRL's docs note that Gemma's attention soft-capping produces
NaN logits unless a flash-attention implementation is used. Pass
--attn-implementation if the loss goes to NaN; this script fails loudly on a
non-finite loss rather than training on it silently.

    uv run --project train python train_gkd.py \
      --student ../models/gemma3-xsum-merged-linear \
      --teacher ../models/gemma3-xsum-merged-linear \
      --dataset ../datasets/xsum_dataset/train \
      --output-dir ../models/gemma3-xsum-kd-online
"""

import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl.experimental.gkd import GKDConfig, GKDTrainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, help="Student init - the merge.")
    parser.add_argument("--teacher", required=True, help="Teacher model path or hub id.")
    parser.add_argument("--dataset", required=True, help="save_to_disk dataset with a messages column.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--messages-column", default="messages")
    # Defaults follow the paper's finding that high on-policy fractions work
    # best; beta 0.5 is TRL's default and is the knob most worth sweeping.
    parser.add_argument("--lmbda", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--attn-implementation", default=None,
                        help='e.g. "kernels-community/flash-attn2" - see the Gemma warning above.')
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def surface_vocab_size(model):
    """Expose config.vocab_size for TRL, which compares it across the pair.

    GKDTrainer guards against a student and teacher with mismatched vocabularies
    by reading `model.config.vocab_size` directly. Gemma 3 is multimodal, so its
    composite Gemma3Config keeps vocab_size on `.text_config` and the top-level
    lookup raises AttributeError before training starts. Copying the value up is
    a no-op numerically - it is the same vocabulary either way - and keeps the
    guard doing what it was written to do.
    """
    config = model.config
    if not hasattr(config, "vocab_size") and hasattr(config, "text_config"):
        config.vocab_size = config.text_config.vocab_size
    return model


class FailOnNonFiniteLoss(TrainerCallback):
    """Stop rather than train through NaN.

    Gemma's soft-capping can produce non-finite logits under some attention
    implementations. A run that NaNs quietly still saves an adapter, and the
    only symptom is a model that generates nothing coherent - which is exactly
    how a colleague's distilled model failed. Better to fail at step 1.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None and not torch.isfinite(torch.tensor(float(loss))):
            raise RuntimeError(
                f"Non-finite loss ({loss}) at step {state.global_step}. For Gemma this is "
                "usually attention soft-capping - retry with "
                '--attn-implementation "kernels-community/flash-attn2".'
            )


def main():
    args = parse_args()

    dataset = load_from_disk(args.dataset)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    # Keep only what the trainer needs; the prepared datasets carry document,
    # summary, prompt and id alongside the messages.
    dataset = dataset.select_columns([args.messages_column])
    if args.messages_column != "messages":
        dataset = dataset.rename_column(args.messages_column, "messages")
    print(f"Training examples: {len(dataset)}")

    model_kwargs = {"dtype": torch.bfloat16}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    student = AutoModelForCausalLM.from_pretrained(args.student, **model_kwargs)
    # The teacher is frozen and used only for scoring; loading it separately
    # (rather than by name through GKDConfig) keeps the dtype and attention
    # implementation identical to the student's, so the two disagree only where
    # their weights do.
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, **model_kwargs)
    teacher.eval()

    surface_vocab_size(student)
    surface_vocab_size(teacher)

    config = GKDConfig(
        output_dir=args.output_dir,
        lmbda=args.lmbda,
        beta=args.beta,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
    )

    trainer = GKDTrainer(
        model=student,
        teacher_model=teacher,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        # LoRA on the student, matching every other arm in this project so the
        # comparison is about the distillation regime and not the capacity.
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
        callbacks=[FailOnNonFiniteLoss()],
    )

    print(f"GKD: lmbda={args.lmbda} (on-policy fraction), beta={args.beta}, "
          f"temperature={args.temperature}")
    trainer.train()

    output = Path(args.output_dir)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
