import argparse
import importlib
import json
from pathlib import Path

import tinker
from tinker import types


ev = importlib.import_module("03_evaluate_checkpoint_losses")
ft = importlib.import_module("02_tinker_fine_tuning")
CONDS = ["demos", "first_person", "sdf"]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def pick_sampler_paths(manifest_dir: Path, run_tag: str, target_n: int, allow_leq: bool):
    lookup = ev.load_checkpoint_paths(manifest_dir, run_tag)
    out = {}
    for cond in CONDS:
        ns = sorted(n for (c, n) in lookup if c == cond)
        if not ns:
            raise ValueError(f"No checkpoints found for {cond}")
        if target_n in ns:
            n = target_n
        elif allow_leq:
            leq = [x for x in ns if x <= target_n]
            if not leq:
                raise ValueError(f"No checkpoint <= {target_n} for {cond}")
            n = leq[-1]
        else:
            raise ValueError(f"Missing exact checkpoint n={target_n} for {cond}")

        manifest_path = manifest_dir / f"checkpoint_manifest_{run_tag}_{cond}.jsonl"
        rows = read_jsonl(manifest_path)
        rec = next((r for r in rows if int(r["examples_seen"]) == n), None)
        if rec is None:
            raise ValueError(f"Could not find sampler path for {cond} n={n} in {manifest_path}")
        out[cond] = rec
    return out


def build_chat_prompt(prompt: str):
    tok, _ = ft._get_tok_and_eos()
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok.encode(text, add_special_tokens=False)
    return types.ModelInput.from_ints(ids), tok


def sample_text(client, model_input, tok, max_tokens: int, temperature: float, seed: int):
    params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed)
    resp = client.sample(prompt=model_input, num_samples=1, sampling_params=params).result()
    seq = resp.sequences[0]
    return tok.decode(seq.tokens).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=Path, required=True, help="JSONL file from 07_generate_probe_prompts.py")
    p.add_argument("--manifest-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--run-tag", type=str, required=True)
    p.add_argument("--target-n", type=int, default=500)
    p.add_argument("--allow-leq", action="store_true")
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=220)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data" / "probe_answers.jsonl")
    args = p.parse_args()

    service = tinker.ServiceClient()
    prompts = read_jsonl(args.prompts.resolve())
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]

    selected = pick_sampler_paths(args.manifest_dir.resolve(), args.run_tag, args.target_n, args.allow_leq)

    clients = {
        "base": service.create_sampling_client(base_model=ft.BASE_MODEL),
    }
    for cond in CONDS:
        rec = selected[cond]
        clients[f"ft_{cond}_n{rec['examples_seen']}"] = service.create_sampling_client(
            model_path=rec["sampler_path"]
        )

    rows = []
    for i, prompt_row in enumerate(prompts):
        model_input, tok = build_chat_prompt(prompt_row["prompt"])
        for model_label, client in clients.items():
            text = sample_text(
                client=client,
                model_input=model_input,
                tok=tok,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=1000 + i,
            )
            rows.append({
                "prompt_id": prompt_row.get("prompt_id", f"p_{i:03d}"),
                "domain": prompt_row.get("domain", "unknown"),
                "topic": prompt_row.get("topic", ""),
                "prompt": prompt_row["prompt"],
                "model_label": model_label,
                "answer": text,
            })

    out = args.out.resolve()
    write_jsonl(out, rows)
    print(f"Wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
