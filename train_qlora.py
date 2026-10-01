"""QLoRA fine-tuning of the student, as a script for an EC2 GPU instance.

The same training as notebook/train.ipynb (cell 3). Checkpoints are copied to S3 after every
save, so a Spot interruption only loses the steps since the last one: run it again on a new
instance and it resumes.

Usage (on the instance):
  python train_qlora.py --s3 s3://$DATA_BUCKET/v1
"""

import argparse
import glob
import os
import subprocess

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def sh(*cmd):
    subprocess.run(cmd, check=True)


class SyncToS3(TrainerCallback):
    """After each checkpoint, mirror the run folder to S3 (the instance's disk is not kept)."""

    def __init__(self, local, remote):
        self.local, self.remote = local, remote

    def on_save(self, args, state, control, **kwargs):
        sh("aws", "s3", "sync", self.local, self.remote, "--delete", "--only-show-errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", required=True, help="dataset version, e.g. s3://my-bucket/v1")
    ap.add_argument("--out", default="run")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16, help="LoRA rank (alpha = 2 * r)")
    args = ap.parse_args()

    name = f"qwen1.5b-lora-r{args.r}"
    runs_s3 = f"{args.s3}/runs/{name}"

    os.makedirs("data", exist_ok=True)
    for f in ("train", "val"):
        sh("aws", "s3", "cp", f"{args.s3}/{f}.jsonl", "data/", "--only-show-errors")
    # An interrupted run left its checkpoints in S3: bring them back to resume
    sh("aws", "s3", "sync", runs_s3, args.out, "--only-show-errors")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        dtype=torch.float16,  # the T4 has no bf16: keep the layers that are not quantized in fp16
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map="auto",
    )
    data = load_dataset(
        "json",
        data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl"},
    )
    peft_config = LoraConfig(
        r=args.r,
        lora_alpha=2 * args.r,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    sft_args = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        max_length=1024,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[SyncToS3(args.out, runs_s3)],
    )
    # LoRA weights in fp32: fp16 mixed precision cannot unscale fp16/bf16 gradients
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    trainer.model.print_trainable_parameters()

    resume = bool(glob.glob(f"{args.out}/checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(f"{args.out}/adapter")
    tokenizer.save_pretrained(f"{args.out}/adapter")
    print("final eval loss:", trainer.evaluate()["eval_loss"])

    sh("aws", "s3", "sync", f"{args.out}/adapter", f"{args.s3}/models/{name}/adapter", "--only-show-errors")
    sh("aws", "s3", "sync", args.out, runs_s3, "--only-show-errors")
    print(f"adapter in {args.s3}/models/{name}/adapter/")


if __name__ == "__main__":
    main()
