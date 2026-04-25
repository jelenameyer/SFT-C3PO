import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = ["formality", "verbosity_fussiness", "anxiety", "deference", "helpfulness"]
PERSONA_METRICS = ["formality", "verbosity_fussiness", "anxiety", "deference"]
VIRIDIS = plt.colormaps["viridis"]


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


def bucket_by_domain(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("domain", "unknown")].append(row)
    return dict(grouped)


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
    colors = [VIRIDIS(i / max(len(labels) - 1, 1)) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color=colors)
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
    colors = [VIRIDIS(i / max(len(METRICS) - 1, 1)) for i in range(len(METRICS))]

    fig, ax = plt.subplots(figsize=(12, 6))
    offsets = [(-2 + i) * width for i in range(len(METRICS))]
    for i, metric in enumerate(METRICS):
        vals = [means[k][metric] for k in labels]
        ax.bar([xi + offsets[i] for xi in x], vals, width=width, label=metric, color=colors[i])

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
    p.add_argument("--judged", type=Path, default="data/probe_judged.jsonl", help="JSONL from 09_judge_c3po_style.py")
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

    # Also emit the same artifacts split by `domain` (in_domain / out_of_domain).
    by_domain = bucket_by_domain(rows)
    for domain, domain_rows in sorted(by_domain.items()):
        domain_slug = domain.replace(" ", "_")
        domain_means = group_means(domain_rows)
        domain_csv = out_prefix.with_name(f"{out_prefix.name}_{domain_slug}.csv")
        domain_overall = out_prefix.with_name(f"{out_prefix.name}_{domain_slug}_overall.png")
        domain_dims = out_prefix.with_name(f"{out_prefix.name}_{domain_slug}_dimensions.png")
        write_summary_csv(domain_csv, domain_means)
        plot_overall(domain_means, domain_overall)
        plot_dimensions(domain_means, domain_dims)
        print(f"Wrote {domain_csv}")
        print(f"Wrote {domain_overall}")
        print(f"Wrote {domain_dims}")


if __name__ == "__main__":
    main()
