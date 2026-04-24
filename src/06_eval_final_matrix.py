import argparse
import csv
import json
import re
from pathlib import Path


CONDS = ["demos", "first_person", "sdf"]
TESTSETS = ["demos", "first_person", "sdf"]
CKPT_RE = re.compile(r"^c3po-(?P<cond>demos|first_person|sdf)-n(?P<n>\d+)$")


def load_json(path: Path):
    return json.loads(path.read_text())


def get_loss_blob(entry: dict):
    # Backward compatibility with nested formats.
    if isinstance(entry, dict) and "losses" in entry and isinstance(entry["losses"], dict):
        return entry["losses"]
    return entry


def get_base_losses(results: dict):
    base_blob = get_loss_blob(results["base"])
    out = {}
    for t in TESTSETS:
        out[t] = float(base_blob[t]["mean_loss_per_trained_token"])
    return out


def available_checkpoints(results: dict, cond: str):
    out = []
    for key in results.keys():
        m = CKPT_RE.match(key)
        if not m:
            continue
        if m.group("cond") != cond:
            continue
        out.append(int(m.group("n")))
    return sorted(out)


def choose_checkpoint(results: dict, cond: str, target_n: int, allow_leq: bool):
    ns = available_checkpoints(results, cond)
    if not ns:
        raise ValueError(f"No checkpoints found in eval JSON for condition={cond}")
    if target_n in ns:
        return target_n
    if allow_leq:
        leq = [n for n in ns if n <= target_n]
        if leq:
            return leq[-1]
    raise ValueError(
        f"No checkpoint for condition={cond} at n={target_n}. "
        f"Available: {ns}. Try --allow-leq."
    )


def get_ft_losses(results: dict, cond: str, n: int):
    key = f"c3po-{cond}-n{n}"
    blob = get_loss_blob(results[key])
    out = {}
    for t in TESTSETS:
        out[t] = float(blob[t]["mean_loss_per_trained_token"])
    return out


def build_delta_table(base_losses: dict, ft_losses_by_cond: dict):
    # Rows: test sets. Columns: base, ft_demos, ft_first_person, ft_sdf.
    table = {}
    for testset in TESTSETS:
        table[testset] = {
            "base": 0.0,
            "ft_demos": ft_losses_by_cond["demos"][testset] - base_losses[testset],
            "ft_first_person": ft_losses_by_cond["first_person"][testset] - base_losses[testset],
            "ft_sdf": ft_losses_by_cond["sdf"][testset] - base_losses[testset],
        }
    return table


def write_csv(path: Path, table: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_set", "base", "ft_demos", "ft_first_person", "ft_sdf"])
        for testset in TESTSETS:
            row = table[testset]
            w.writerow([testset, row["base"], row["ft_demos"], row["ft_first_person"], row["ft_sdf"]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", type=Path, required=True, help="Path to eval_raw*.json from 03_eval_base_model.py")
    p.add_argument("--target-n", type=int, default=500, help="Target final checkpoint n per condition.")
    p.add_argument("--allow-leq", action="store_true",
                   help="If exact target-n missing, use largest checkpoint <= target-n per condition.")
    p.add_argument("--out-csv", type=Path, default=None, help="Output CSV path.")
    p.add_argument("--out-json", type=Path, default=None, help="Output JSON path.")
    args = p.parse_args()

    eval_json = args.eval_json.resolve()
    results = load_json(eval_json)

    base_losses = get_base_losses(results)
    chosen = {cond: choose_checkpoint(results, cond, args.target_n, args.allow_leq) for cond in CONDS}
    ft_losses_by_cond = {cond: get_ft_losses(results, cond, chosen[cond]) for cond in CONDS}
    table = build_delta_table(base_losses, ft_losses_by_cond)

    out_csv = (
        args.out_csv.resolve()
        if args.out_csv
        else eval_json.with_name(f"{eval_json.stem}_delta_table_n{args.target_n}.csv")
    )
    out_json = (
        args.out_json.resolve()
        if args.out_json
        else eval_json.with_name(f"{eval_json.stem}_delta_table_n{args.target_n}.json")
    )

    write_csv(out_csv, table)
    payload = {
        "source_eval_json": str(eval_json),
        "target_n": args.target_n,
        "chosen_checkpoint_n": chosen,
        "base_losses": base_losses,
        "delta_table": table,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print("Delta table (rows=test sets; cols=base, ft_demos, ft_first_person, ft_sdf):")
    for t in TESTSETS:
        r = table[t]
        print(
            f"{t:12s} | {r['base']:8.4f} | {r['ft_demos']:8.4f} | "
            f"{r['ft_first_person']:8.4f} | {r['ft_sdf']:8.4f}"
        )


if __name__ == "__main__":
    main()
