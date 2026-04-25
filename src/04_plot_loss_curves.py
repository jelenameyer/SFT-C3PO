import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


CKPT_RE = re.compile(r"^c3po-(?P<cond>demos|first_person|sdf)-n(?P<n>\d+)$")
CONDS = ["demos", "first_person", "sdf"]


def load_eval(path: Path):
    return json.loads(path.read_text())


def get_base_losses(results: dict):
    base = results.get("base", {})
    # Backward compatibility: some formats store losses under base["losses"].
    if "losses" in base and isinstance(base["losses"], dict):
        return base["losses"]
    return base


def collect_points(results: dict, test_set: str):
    """Return dict[train_cond] -> sorted list[(examples_seen, loss)]."""
    out = {c: [] for c in CONDS}
    for key, val in results.items():
        m = CKPT_RE.match(key)
        if not m:
            continue
        train_cond = m.group("cond")
        n = int(m.group("n"))
        losses_blob = val.get("losses", val)
        if test_set not in losses_blob:
            continue
        loss = losses_blob[test_set]["mean_loss_per_trained_token"]
        out[train_cond].append((n, loss))
    for c in CONDS:
        out[c].sort(key=lambda x: x[0])
    return out


def plot_diagonal(results: dict, out_path: Path):
    base_losses = get_base_losses(results)
    fig, ax = plt.subplots(figsize=(9, 5))

    for cond in CONDS:
        points = collect_points(results, test_set=cond)[cond]
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", label=f"{cond} -> {cond}")
        if cond in base_losses:
            yb = base_losses[cond]["mean_loss_per_trained_token"]
            ax.axhline(y=yb, linestyle="--", alpha=0.35, label=f"base on {cond}")

    ax.set_title("Sample Efficiency (Diagonal: train cond -> same test cond)")
    ax.set_xlabel("Examples Seen")
    ax.set_ylabel("Mean Loss Per Trained Token")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_per_testset(results: dict, out_prefix: Path):
    base_losses = get_base_losses(results)
    for test_set in CONDS:
        curves = collect_points(results, test_set=test_set)
        fig, ax = plt.subplots(figsize=(9, 5))

        any_points = False
        for train_cond in CONDS:
            points = curves[train_cond]
            if not points:
                continue
            any_points = True
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", label=f"{train_cond} -> {test_set}")

        if test_set in base_losses:
            yb = base_losses[test_set]["mean_loss_per_trained_token"]
            ax.axhline(y=yb, linestyle="--", alpha=0.35, label=f"base on {test_set}")

        ax.set_title(f"Loss Curves on Test Set: {test_set}")
        ax.set_xlabel("Examples Seen")
        ax.set_ylabel("Mean Loss Per Trained Token")
        ax.grid(True, alpha=0.25)
        if any_points or test_set in base_losses:
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_prefix.with_name(f"{out_prefix.name}_{test_set}.png"), dpi=160)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", type=Path, default="data/outputs/eval_loss_raw_final_20260425.json", help="Path to eval JSON from 03_eval_base_model.py")
    p.add_argument("--out-prefix", type=Path, default=None,
                   help="Prefix for output PNG files.")
    args = p.parse_args()

    eval_json = args.eval_json.resolve()
    results = load_eval(eval_json)
    if args.out_prefix:
        out_prefix = args.out_prefix.resolve()
    else:
        out_dir = eval_json.parent if eval_json.parent.name == "outputs" else (eval_json.parent / "outputs")
        out_prefix = out_dir / "loss_curves"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    diagonal_path = out_prefix.with_name(f"{out_prefix.name}_diagonal.png")
    plot_diagonal(results, diagonal_path)
    plot_per_testset(results, out_prefix)
    print(f"Wrote {diagonal_path}")
    print(f"Wrote {out_prefix.with_name(f'{out_prefix.name}_demos.png')}")
    print(f"Wrote {out_prefix.with_name(f'{out_prefix.name}_first_person.png')}")
    print(f"Wrote {out_prefix.with_name(f'{out_prefix.name}_sdf.png')}")


if __name__ == "__main__":
    main()
