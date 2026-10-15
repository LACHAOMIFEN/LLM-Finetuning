"""Single-A40 Qwen LoRA SFT, with prompt masking and before/after evaluation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(Path("data/hf_cache").resolve()))

from prepare_tnews import LABELS, read_jsonl


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def tokenize_example(tokenizer, row, max_length):
    prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError("Chat template prefix mismatch; cannot safely mask prompt")
    if len(full) > max_length:
        raise ValueError(f'{row["id"]}: {len(full)} tokens exceeds {max_length}; do not silently truncate answers')
    labels = [-100] * len(prompt) + full[len(prompt):]
    assert any(x != -100 for x in labels)
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


def score_predictions(rows):
    valid, correct, per_class = 0, 0, {}
    for category in LABELS.values():
        per_class[category] = {"support": 0, "correct": 0}
    for row in rows:
        try:
            obj = json.loads(row["answer"])
            compliant = isinstance(obj, dict) and set(obj) == {"category"} and obj["category"] in LABELS.values()
        except (ValueError, TypeError):
            obj, compliant = {}, False
        pred = obj.get("category") if compliant else None
        row.update(prediction=pred, format_valid=compliant, correct=pred == row["category"])
        valid += compliant
        correct += row["correct"]
        per_class[row["category"]]["support"] += 1
        per_class[row["category"]]["correct"] += row["correct"]
    for category, stats in per_class.items():
        tp = stats["correct"]
        predicted = sum(r["prediction"] == category for r in rows)
        stats["f1"] = 2 * tp / (predicted + stats["support"]) if predicted + stats["support"] else 0
    return {"n": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "format_valid_rate": valid / len(rows),
            "macro_f1": sum(s["f1"] for s in per_class.values()) / len(per_class), "per_class": per_class}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", default="data/tnews")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new, empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "status.json", {"phase": "loading", "started": time.time()})

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Trainer, TrainerCallback, TrainingArguments, set_seed)

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Run inside the allocated A40 job with BF16 support")
    torch.set_num_threads(8)
    set_seed(42)
    model_path = str(Path(args.model).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    datasets = {name: read_jsonl(Path(args.data) / f"{name}.jsonl") for name in ["train", "validation", "test"]}
    if args.smoke:
        datasets = {k: v[:32 if k == "train" else 8] for k, v in datasets.items()}
    encoded = {key: [tokenize_example(tokenizer, row, 256) for row in rows]
               for key, rows in datasets.items() if key != "test"}
    for split, rows in datasets.items():
        for row in rows:
            assert len(tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True, add_generation_prompt=True)) <= 256
    token_stats = {key: {"max": max(len(x["input_ids"]) for x in rows),
                         "mean": sum(len(x["input_ids"]) for x in rows) / len(rows)} for key, rows in encoded.items()}
    for key, rows in datasets.items():
        (output / f"{key}_data.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True, use_safetensors=True,
        ).to("cuda")

    model = get_peft_model(load_base(), LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], bias="none",
    ))
    model.config.use_cache = False
    model.print_trainable_parameters()
    config = {**vars(args), "model_path": model_path, "seed": 42,
              "batch_size": 4, "gradient_accumulation_steps": 4, "learning_rate": 2e-4,
              "max_length": 256, "loss_scope": "assistant_answer_and_end_tokens_only",
              "gpu": torch.cuda.get_device_name(), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
              "token_stats": token_stats, "split_sizes": {k: len(v) for k, v in datasets.items()},
              "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "datasets", "accelerate"]},
              "model_source": json.loads((Path(model_path) / "download_manifest.json").read_text()),
              "data_sha256": {k: hashlib.sha256((output / f"{k}_data.jsonl").read_bytes()).hexdigest() for k in datasets},
              "selection_rule": "highest validation accuracy; earliest checkpoint on ties",
              "initialization": "original base model with fresh LoRA; not a continuation of the previous one-epoch adapter"}
    save_json(output / "run_config.json", config)

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            entry = {**(logs or {}), "step": state.global_step, "epoch": state.epoch, "timestamp": time.time()}
            with (output / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            save_json(output / "status.json", {"phase": "training_or_validation", "step": state.global_step,
                      "max_steps": state.max_steps, "logs": logs, "updated": time.time()})

    class ValidationTrainer(Trainer):
        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            phase = f"{metric_key_prefix}_validation_step_{self.state.global_step:06d}"
            stats, _ = evaluate_generation(self.model, phase, split="validation")
            extra = {f"{metric_key_prefix}_{k}": stats[k] for k in ["accuracy", "macro_f1", "format_valid_rate"]}
            metrics.update(extra)
            self.log(extra)
            save_json(output / f"{phase}_summary.json", {
                "step": self.state.global_step, "epoch": self.state.epoch or 0, **metrics,
            })
            return metrics

    trainer = ValidationTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=Dataset.from_list(encoded["train"]),
        eval_dataset=Dataset.from_list(encoded["validation"]),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100),
        args=TrainingArguments(
            output_dir=str(output), num_train_epochs=args.epochs, max_steps=2 if args.smoke else -1,
            per_device_train_batch_size=4, per_device_eval_batch_size=4, gradient_accumulation_steps=4,
            learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
            bf16=True, tf32=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch", logging_steps=1 if args.smoke else 5, save_strategy="epoch",
            save_total_limit=max(1, args.epochs), eval_strategy="epoch", report_to="none",
            load_best_model_at_end=True, metric_for_best_model="accuracy", greater_is_better=True,
            seed=42, data_seed=42, dataloader_num_workers=0, disable_tqdm=True,
            label_names=["labels"], prediction_loss_only=True,
        ), callbacks=[Progress()],
    )

    def evaluate_generation(current_model, phase, split="test"):
        save_json(output / "status.json", {"phase": phase, "updated": time.time()})
        was_training = current_model.training
        checkpointing = current_model.is_gradient_checkpointing
        previous_padding = tokenizer.padding_side
        current_model.eval()
        current_model.gradient_checkpointing_disable()
        tokenizer.padding_side = "left"
        predictions = []
        rows = datasets[split]
        start = time.monotonic()
        for offset in range(0, len(rows), 16):
            batch = rows[offset:offset + 16]
            prompts = [tokenizer.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True) for r in batch]
            inputs = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = current_model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True,
                                                   pad_token_id=tokenizer.pad_token_id)
            answers = tokenizer.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            predictions.extend({"id": r["id"], "title": r["title"], "category": r["category"], "answer": a}
                               for r, a in zip(batch, answers))
            if len(predictions) == len(rows):
                print(f"{phase}: {len(predictions)}/{len(rows)}", flush=True)
        metrics = score_predictions(predictions)
        metrics["seconds"] = time.monotonic() - start
        save_json(output / f"{phase}_predictions.json", predictions)
        save_json(output / f"{phase}_metrics.json", metrics)
        tokenizer.padding_side = previous_padding
        if checkpointing:
            current_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        current_model.train(was_training)
        return metrics, predictions

    before, baseline = evaluate_generation(model, "before")
    print("BEFORE", json.dumps(before, ensure_ascii=False), flush=True)
    before_loss = trainer.evaluate(metric_key_prefix="before")
    save_json(output / "before_validation.json", before_loss)
    save_json(output / "status.json", {"phase": "training", "updated": time.time()})
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    selection = {"best_checkpoint": trainer.state.best_model_checkpoint,
                 "best_validation_accuracy": trainer.state.best_metric,
                 "criterion": config["selection_rule"]}
    checkpoint_state = Path(trainer.state.best_model_checkpoint) / "trainer_state.json"
    selection["best_epoch"] = json.loads(checkpoint_state.read_text())["epoch"]
    save_json(output / "selection.json", selection)
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(str(output / "adapter"))
    trainer.save_state()
    save_json(output / "train_metrics.json", {**result.metrics, "peak_allocated_gib": peak_gib})
    after_loss = trainer.evaluate(metric_key_prefix="after")
    save_json(output / "after_validation.json", after_loss)
    # Evaluate the saved adapter after loading it again, which also verifies reload.
    del trainer, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(load_base(), str(output / "adapter"), local_files_only=True)
    after, tuned = evaluate_generation(model, "after")
    save_json(output / "comparison.json", [
        {"id": a["id"], "title": a["title"], "gold": a["category"], "before": a["answer"],
         "after": b["answer"], "before_correct": a["correct"], "after_correct": b["correct"]}
        for a, b in zip(baseline, tuned)
    ])
    summary = {"before": before, "after": after, "validation_before": before_loss,
               "validation_after": after_loss, "training": result.metrics, "peak_allocated_gib": peak_gib,
               "accuracy_change_percentage_points": 100 * (after["accuracy"] - before["accuracy"]),
               "adapter_reload_verified": True, "selection": selection}
    save_json(output / "metrics.json", summary)
    from plot_tnews import plot_run
    plot_run(output)
    save_json(output / "status.json", {"phase": "complete", "updated": time.time()})
    print("COMPLETE", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
