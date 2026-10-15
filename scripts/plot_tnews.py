"""Export recorded training curves to PNG/PDF and tidy CSV; never fabricate points."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/matplotlib_cache").resolve()))


def plot_run(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines() if line.strip()]
    # Merge loss and generation metrics logged separately at the same step.
    validation = {}
    for row in history:
        for prefix in ["before", "eval"]:
            if any(key.startswith(prefix + "_") for key in row):
                entry = validation.setdefault(row["step"], {"step": row["step"], "epoch": row.get("epoch") or 0})
                for key in ["loss", "accuracy", "macro_f1", "format_valid_rate"]:
                    if prefix + "_" + key in row:
                        entry[key] = row[prefix + "_" + key]
    train = [r for r in history if "loss" in r and "learning_rate" in r]
    val = [validation[k] for k in sorted(validation)]
    if not train or not val:
        raise ValueError("No recorded train/validation values to plot")
    with (output / "training_history.csv").open("w", newline="") as stream:
        fields = ["step", "epoch", "loss", "learning_rate", "grad_norm", "timestamp"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(train)
    with (output / "validation_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "epoch", "loss", "accuracy", "macro_f1", "format_valid_rate"])
        writer.writeheader()
        writer.writerows(val)
    selection_path = output / "selection.json"
    selected = json.loads(selection_path.read_text())["best_epoch"] if selection_path.exists() else None
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = [r["epoch"] for r in train]
    vx = [r["epoch"] for r in val]
    axes[0, 0].plot(x, [r["loss"] for r in train], label="Train loss (logging-window mean)", color="#2563eb")
    axes[0, 0].plot(vx, [r["loss"] for r in val], "o-", label="Validation loss", color="#ea580c")
    axes[0, 0].set(title="Answer-token loss", ylabel="Cross entropy")
    axes[0, 0].legend(fontsize=8)
    for key, label, color in [("accuracy", "Accuracy", "#2563eb"), ("macro_f1", "Macro-F1", "#7c3aed"),
                              ("format_valid_rate", "Valid format / category", "#16a34a")]:
        axes[0, 1].plot(vx, [r[key] * 100 for r in val], "o-", label=label, color=color)
    axes[0, 1].set(title="Validation set only", ylabel="Percent", ylim=(0, 105))
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(x, [r["learning_rate"] for r in train], color="#0891b2")
    axes[1, 0].set(title="Learning rate", ylabel="Learning rate")
    axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    grad_rows = [r for r in train if "grad_norm" in r]
    axes[1, 1].plot([r["epoch"] for r in grad_rows], [r["grad_norm"] for r in grad_rows], color="#7c3aed")
    axes[1, 1].set(title="Gradient norm (before clipping)", ylabel="L2 norm")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        if selected is not None:
            ax.axvline(selected, color="#64748b", linestyle="--", alpha=0.55)
    fig.suptitle("Qwen2.5-1.5B LoRA / TNEWS" + (f" — selected epoch: {selected:g}" if selected is not None else ""))
    fig.savefig(output / "training_curves.png", dpi=180)
    fig.savefig(output / "training_curves.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(x, [r["loss"] for r in train], label="Train loss (logging-window mean)")
    ax.plot(vx, [r["loss"] for r in val], "o-", label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Cross entropy", title="Training and validation loss")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output / "loss_curve.png", dpi=180)
    plt.close(fig)
    print(f"Saved curves and CSV logs to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    plot_run(parser.parse_args().output)
