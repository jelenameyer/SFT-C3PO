import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


CONDS = ["demos", "first_person", "sdf"]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--run-tag", type=str, required=True)
    p.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    args = p.parse_args()

    log_dir = args.log_dir.resolve()
    out = (
        args.out.resolve()
        if args.out
        else (log_dir / "outputs" / f"train_loss_curves_{args.run_tag}.png")
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    found = False

    for cond in CONDS:
        path = log_dir / f"train_log_{args.run_tag}_{cond}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        if not rows:
            continue
        xs = [r["examples_seen"] for r in rows]
        ys = [r["loss"] for r in rows]
        ax.plot(xs, ys, marker="o", label=cond)
        found = True

    if not found:
        raise FileNotFoundError(f"No train logs found for run_tag={args.run_tag} in {log_dir}")

    ax.set_title(f"Fine-Tuning Loss Curves ({args.run_tag})")
    ax.set_xlabel("Examples Seen")
    ax.set_ylabel("Training Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
