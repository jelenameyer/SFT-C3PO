# eval_loss.py  (skeleton — extend for your full matrix)
import json, importlib
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

def eval_state(state_name, test_sets):
    """state_name=None => baseline Qwen."""
    service = tinker.ServiceClient()
    client = service.create_lora_training_client(base_model=ft.BASE_MODEL, rank=ft.LORA_RANK)
    if state_name is not None:
        client.load_state(name=state_name).result()   # VERIFY API

    out = {}
    for tname, examples in test_sets.items():
        losses, ntoks = [], []
        for (inp, tgt, w), idx in examples:
            batch = [ft.make_datum(inp, tgt, w)]
            # FORWARD-ONLY. VERIFY your Tinker version's signature:
            #   preferred: client.forward(batch, loss_fn="cross_entropy")
            #   fallback : client.forward_backward(batch, loss_fn="cross_entropy", do_backward=False)
            #   fallback2: forward_backward and never call optim_step (watch grad accumulation)
            fb = client.forward_backward(batch, loss_fn="cross_entropy")
            loss = float(fb.result().loss)            # mean over weighted tokens in the batch
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

if __name__ == "__main__":
    test_sets = {c: load_test(c) for c in CONDS}
    results = {"base": eval_state(None, test_sets)}
    for c in CONDS:
        for n in CKPTS:
            results[f"c3po-{c}-n{n}"] = eval_state(f"c3po-{c}-n{n}", test_sets)
    (DATA / "eval_loss_raw.json").write_text(json.dumps(results, indent=2))