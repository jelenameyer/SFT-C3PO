# eval_loss.py
import argparse
import importlib
import json
from pathlib import Path

import tinker

ft = importlib.import_module("02_tinker_fine_tuning")  # render_demo, render_text, make_datum, BASE_MODEL, LORA_RANK

DATA = Path(__file__).resolve().parent / "data"
CONDS = ["demos", "first_person", "sdf"]
CKPTS = [100, 200, 300, 400, 500]

def load_test(cond):
    rows = [json.loads(l) for l in (DATA / f"{cond}_test.jsonl").read_text().splitlines() if l.strip()]
    if cond == "demos":
        return [(ft.render_demo(r["user"], r["assistant"]), i) for i, r in enumerate(rows)]
    return [(ft.render_text(r["text"]), i) for i, r in enumerate(rows)]

def _batch_loss(client, batch):
    """Forward-only loss if available; fallback to forward_backward without optim step."""
    if hasattr(client, "forward"):
        return float(client.forward(batch, loss_fn="cross_entropy").result().loss)
    return float(client.forward_backward(batch, loss_fn="cross_entropy").result().loss)


def eval_state(state_path, test_sets):
    """state_path=None => baseline Qwen."""
    service = tinker.ServiceClient()
    client = service.create_lora_training_client(base_model=ft.BASE_MODEL, rank=ft.LORA_RANK)
    if state_path is not None:
        client.load_state(state_path).result()

    out = {}
    for tname, examples in test_sets.items():
        losses, ntoks = [], []
        for (inp, tgt, w), idx in examples:
            batch = [ft.make_datum(inp, tgt, w)]
            loss = _batch_loss(client, batch)         # mean over weighted tokens in the batch
            n    = int(sum(1 for x in w if x > 0))
            losses.append(loss); ntoks.append(n)
        # token-weighted mean across the test set
        total_nll = sum(l * n for l, n in zip(losses, ntoks))
        out[tname] = {
            "mean_loss_per_trained_token": total_nll / sum(ntoks),
            "n_tokens": sum(ntoks),
            "per_example": [{"loss": l, "n_trained_tokens": n} for l, n in zip(losses, ntoks)],
        }
    return out


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_checkpoint_paths(manifest_dir: Path, run_tag: str | None):
    """
    Returns:
      dict[(cond, n_examples)] = {"state_path": ..., "checkpoint_name": ...}
    """
    lookup = {}
    for cond in CONDS:
        if run_tag:
            manifest_paths = [manifest_dir / f"checkpoint_manifest_{run_tag}_{cond}.jsonl"]
        else:
            manifest_paths = sorted(manifest_dir.glob(f"checkpoint_manifest_*_{cond}.jsonl"))
            if manifest_paths:
                manifest_paths = [manifest_paths[-1]]  # newest lexicographically (timestamp-style run tags)

        for mpath in manifest_paths:
            for rec in _read_jsonl(mpath):
                n = int(rec.get("examples_seen", -1))
                if n in CKPTS and "state_path" in rec:
                    lookup[(cond, n)] = {
                        "state_path": rec["state_path"],
                        "checkpoint_name": rec.get("checkpoint_name", f"{cond}-n{n}"),
                    }
    return lookup


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=DATA,
                        help="Directory containing checkpoint_manifest_*.jsonl files.")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="If set, only use checkpoint manifests for this run tag.")
    parser.add_argument("--out", type=Path, default=DATA / "eval_loss_raw.json",
                        help="Output JSON path.")
    args = parser.parse_args()

    test_sets = {c: load_test(c) for c in CONDS}
    results = {"base": eval_state(None, test_sets)}
    ckpt_lookup = load_checkpoint_paths(args.manifest_dir.resolve(), args.run_tag)

    missing = []
    for c in CONDS:
        for n in CKPTS:
            key = (c, n)
            label = f"c3po-{c}-n{n}"
            if key not in ckpt_lookup:
                missing.append(label)
                continue
            rec = ckpt_lookup[key]
            results[label] = {
                "meta": {
                    "state_path": rec["state_path"],
                    "checkpoint_name": rec["checkpoint_name"],
                },
                "losses": eval_state(rec["state_path"], test_sets),
            }

    if missing:
        results["missing_checkpoints"] = missing

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")
