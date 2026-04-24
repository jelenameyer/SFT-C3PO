import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = ["formality", "verbosity_fussiness", "anxiety", "deference", "helpfulness"]
PERSONA_METRICS = ["formality", "verbosity_fussiness", "anxiety", "deference"]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def group_means(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["model_label"]].append(r)

    means = {}
    for model, rs in grouped.items():
        m = {}
        for metric in METRICS:
            m[metric] = sum(float(x[metric]) for x in rs) / len(rs)
        m["overall_c3po_like"] = sum(m[k] for k in PERSONA_METRICS) / len(PERSONA_METRICS)
        means[model] = m
    return means


def write_summary_csv(path: Path, means: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_label", "overall_c3po_like", *METRICS])
        for model, m in sorted(means.items()):
            w.writerow([model, m["overall_c3po_like"], *[m[k] for k in METRICS]])


def plot_overall(means: dict, out: Path):
    labels = sorted(means.keys())
    values = [means[k]["overall_c3po_like"] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Score (0-10)")
    ax.set_title("Overall C-3PO-Like Score")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_dimensions(means: dict, out: Path):
    labels = sorted(means.keys())
    width = 0.15
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(12, 6))
    offsets = [(-2 + i) * width for i in range(len(METRICS))]
    for i, metric in enumerate(METRICS):
        vals = [means[k][metric] for k in labels]
        ax.bar([xi + offsets[i] for xi in x], vals, width=width, label=metric)

    ax.set_ylim(0, 10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel("Score (0-10)")
    ax.set_title("Judge Scores by Dimension")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judged", type=Path, required=True, help="JSONL from 09_judge_c3po_style.py")
    p.add_argument("--out-prefix", type=Path, default=None)
    args = p.parse_args()

    judged = args.judged.resolve()
    rows = read_jsonl(judged)
    means = group_means(rows)
    if args.out_prefix:
        out_prefix = args.out_prefix.resolve()
    else:
        base_dir = judged.parent if judged.parent.name == "outputs" else (judged.parent / "outputs")
        out_prefix = base_dir / "judge_summary"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_prefix.with_name(f"{out_prefix.name}.csv")
    overall_path = out_prefix.with_name(f"{out_prefix.name}_overall.png")
    dims_path = out_prefix.with_name(f"{out_prefix.name}_dimensions.png")

    write_summary_csv(csv_path, means)
    plot_overall(means, overall_path)
    plot_dimensions(means, dims_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {overall_path}")
    print(f"Wrote {dims_path}")


if __name__ == "__main__":
    main()
