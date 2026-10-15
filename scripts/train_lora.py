"""Small, single-GPU LoRA exercise adapted from the official TRL SFT tutorial."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--output", default="outputs/qwen_lora_50steps")
    args = parser.parse_args()
    if min(args.steps, args.samples, args.eval_samples) < 1:
        parser.error("steps, samples, and eval-samples must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; choose a new --output path.")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check your environment and GPU allocation.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This exercise expects a GPU with BF16 support, such as A40.")
    set_seed(42)
    output.mkdir(parents=True, exist_ok=True)

    def save_json(name, value):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_safetensors=True,
    )
    model.config.use_cache = False

    # Freeze the base weights and train only the small LoRA matrices.
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    dataset = load_dataset("trl-lib/Capybara", split="train")
    dataset = dataset.select_columns(["messages"])
    # Remove identical conversations before drawing disjoint train/eval subsets.
    seen = set()

    def unique(example):
        key = json.dumps(example["messages"], sort_keys=True, ensure_ascii=False)
        if key in seen:
            return False
        seen.add(key)
        return True

    dataset = dataset.filter(unique, load_from_cache_file=False).shuffle(seed=42)
    total = args.samples + args.eval_samples
    if total > len(dataset):
        raise ValueError(f"Requested {total} examples, but only {len(dataset)} are available.")
    train_data = dataset.select(range(args.samples))
    eval_data = dataset.select(range(args.samples, total))
    train_data.to_json(str(output / "train_data.jsonl"), force_ascii=False)
    eval_data.to_json(str(output / "eval_data.jsonl"), force_ascii=False)
    save_json("run_config.json", {
        **vars(args), "model": model_id, "dataset": "trl-lib/Capybara",
        "seed": 42, "max_length": 512, "loss_scope": "full_conversation",
        "gpu": torch.cuda.get_device_name(0),
    })

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=lora,
        args=SFTConfig(
            output_dir=str(output),
            max_steps=args.steps,
            max_length=512,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            warmup_ratio=0.1,
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch",
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            seed=42,
            packing=False,
            completion_only_loss=False,
            assistant_only_loss=False,
            eos_token="<|im_end|>",
        ),
    )
    trainer.model.print_trainable_parameters()

    prompts = [
        "Explain why the sky is blue in two short sentences.",
        "Give three practical tips for learning Python.",
        "请用两句话解释什么是机器学习。",
    ]

    def answers():
        trainer.model.eval()
        results = []
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            inputs = inputs.to(trainer.model.device)
            with torch.inference_mode():
                tokens = trainer.model.generate(
                    **inputs, max_new_tokens=128, do_sample=False,
                    use_cache=True, pad_token_id=tokenizer.pad_token_id,
                )
            answer = tokenizer.decode(
                tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            results.append({"prompt": prompt, "answer": answer})
        return results

    before = trainer.evaluate(metric_key_prefix="before")
    save_json("before_metrics.json", before)
    baseline_answers = answers()
    save_json("before_answers.json", baseline_answers)

    result = trainer.train()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(str(output / "adapter"))
    trainer.save_state()
    after = trainer.evaluate(metric_key_prefix="after")
    save_json("metrics.json", {"before": before, "train": result.metrics, "after": after})
    tuned_answers = answers()
    save_json("comparison.json", [
        {"prompt": a["prompt"], "before": a["answer"], "after": b["answer"]}
        for a, b in zip(baseline_answers, tuned_answers)
    ])
    print(f"Done. Results and adapter saved to {output.resolve()}")


if __name__ == "__main__":
    main()
