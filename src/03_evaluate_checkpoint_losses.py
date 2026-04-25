# eval_loss.py
import argparse
import importlib
import json
from pathlib import Path
import time
import numpy as np
import tinker

ft = importlib.import_module("02_tinker_fine_tuning")  # render_demo, render_text, make_datum, BASE_MODEL, LORA_RANK

DATA = Path(__file__).resolve().parent / "data"
CONDS = ["demos", "first_person", "sdf"]

def load_test(cond, data_dir: Path):
    rows = [json.loads(l) for l in (data_dir / f"{cond}_test.jsonl").read_text().splitlines() if l.strip()]
    if cond == "demos":
        return [(ft.render_demo(r["user"], r["assistant"]), i) for i, r in enumerate(rows)]
    return [(ft.render_text(r["text"]), i) for i, r in enumerate(rows)]

def _to_numpy(tensor_data) -> np.ndarray:
    if hasattr(tensor_data, "to_numpy"):
        return np.asarray(tensor_data.to_numpy())
    if hasattr(tensor_data, "tolist"):
        return np.asarray(tensor_data.tolist())
    return np.asarray(tensor_data)


def _batch_loss(client, batch) -> tuple[float, int]:
    """Returns (mean_nll_per_weighted_token, n_weighted_tokens)."""
    if hasattr(client, "forward"):
        out = client.forward(batch, loss_fn="cross_entropy").result()
    else:
        out = client.forward_backward(batch, loss_fn="cross_entropy").result()

    if not hasattr(out, "loss_fn_outputs") or not out.loss_fn_outputs:
        raise RuntimeError(f"No loss_fn_outputs on {type(out).__name__}")

    total_nll = 0.0
    total_w = 0.0
    n_weighted_tokens = 0
    for ex_out, datum in zip(out.loss_fn_outputs, batch):
        if "elementwise_loss" not in ex_out:
            raise RuntimeError(
                f"Expected key 'elementwise_loss' in {list(ex_out.keys())}"
            )
        per_tok = _to_numpy(ex_out["elementwise_loss"]).astype(np.float64).ravel()
        w = _to_numpy(datum.loss_fn_inputs["weights"]).astype(np.float64).ravel()
        n = min(len(per_tok), len(w))
        per_tok, w = per_tok[:n], w[:n]
        total_nll += float((per_tok * w).sum())
        total_w += float(w.sum())
        n_weighted_tokens += int((w > 0).sum())

    if total_w <= 0:
        raise ValueError("Zero total weight in batch")
    return total_nll / total_w, n_weighted_tokens


def eval_state(
    state_path,
    test_sets,
    model_label: str,
    progress_every: int,
    eval_batch_size: int,
):
    """state_path=None => baseline Qwen."""
    service = tinker.ServiceClient()
    client = service.create_lora_training_client(base_model=ft.BASE_MODEL, rank=ft.LORA_RANK)
    if state_path is not None:
        client.load_state(state_path).result()

    out = {}
    print(f"[eval] model={model_label} | start")
    for tname, examples in test_sets.items():
        print(f"[eval] model={model_label} | test_set={tname} | n_examples={len(examples)}")
        if eval_batch_size <= 0:
            raise ValueError("--eval-batch-size must be > 0")

        total_nll = 0.0
        total_ntoks = 0
        per_example = []
        t0 = time.time()
        for i in range(0, len(examples), eval_batch_size):
            chunk = examples[i : i + eval_batch_size]
            batch = [ft.make_datum(inp, tgt, w) for (inp, tgt, w), _ in chunk]
            batch_loss, batch_ntoks = _batch_loss(client, batch)

            total_nll += batch_loss * batch_ntoks
            total_ntoks += batch_ntoks
            per_example.append({"loss": batch_loss, "n_trained_tokens": batch_ntoks})

            done = min(i + len(chunk), len(examples))
            if progress_every > 0 and (done % progress_every == 0 or done == len(examples)):
                elapsed = time.time() - t0
                print(
                    f"[eval] model={model_label} | test_set={tname} | "
                    f"done={done}/{len(examples)} | elapsed={elapsed:.1f}s"
                )

        out[tname] = {
            "mean_loss_per_trained_token": total_nll / total_ntoks,
            "n_tokens": total_ntoks,
            "per_example": per_example,
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
                if n > 0 and "state_path" in rec:
                    lookup[(cond, n)] = {
                        "state_path": rec["state_path"],
                        "checkpoint_name": rec.get("checkpoint_name", f"{cond}-n{n}"),
                    }
    return lookup


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA,
                        help="Directory containing {cond}_test.jsonl files.")
    parser.add_argument("--manifest-dir", type=Path, default=DATA,
                        help="Directory containing checkpoint_manifest_*.jsonl files.")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="If set, only use checkpoint manifests for this run tag.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path.")
    parser.add_argument("--max-checkpoint-examples", type=int, default=None,
                        help="If set, only evaluate checkpoints with examples_seen <= this value.")
    parser.add_argument("--progress-every", type=int, default=5,
                        help="Print progress every N examples within each test set (0 disables).")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of examples per forward pass during eval. Increase for speed.")
    args = parser.parse_args()

    test_sets = {c: load_test(c, args.data_dir.resolve()) for c in CONDS}
    out_path = (
        args.out.resolve()
        if args.out
        else (args.data_dir.resolve() / "outputs" / "eval_loss_raw.json")
    )

    total_models = 1
    ckpt_lookup = load_checkpoint_paths(args.manifest_dir.resolve(), args.run_tag)

    selected = []
    for (cond, n), rec in ckpt_lookup.items():
        if args.max_checkpoint_examples is not None and n > args.max_checkpoint_examples:
            continue
        selected.append((cond, n, rec))
    selected.sort(key=lambda x: (x[0], x[1]))
    total_models += len(selected)

    print(f"[eval] starting total_models={total_models} (including base)")
    results = {
        "base": eval_state(
            None,
            test_sets,
            model_label="base",
            progress_every=args.progress_every,
            eval_batch_size=args.eval_batch_size,
        )
    }

    for i, (c, n, rec) in enumerate(selected, start=1):
        label = f"c3po-{c}-n{n}"
        print(f"[eval] checkpoint_model {i}/{len(selected)}: {label}")
        results[label] = {
            "meta": {
                "state_path": rec["state_path"],
                "checkpoint_name": rec["checkpoint_name"],
                "examples_seen": n,
            },
            "losses": eval_state(
                rec["state_path"],
                test_sets,
                model_label=label,
                progress_every=args.progress_every,
                eval_batch_size=args.eval_batch_size,
            ),
        }
    results["evaluated_checkpoints"] = [f"c3po-{c}-n{n}" for c, n, _ in selected]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")
