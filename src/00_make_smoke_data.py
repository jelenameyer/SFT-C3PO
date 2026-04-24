import argparse
import json
import random
from pathlib import Path

CONDS = ["demos", "first_person", "sdf"]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def sample_rows(rows, n: int, rng: random.Random):
    if n >= len(rows):
        return list(rows)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    idxs = idxs[:n]
    return [rows[i] for i in idxs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data_smoke")
    p.add_argument("--train-n", type=int, default=10)
    p.add_argument("--test-n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.train_n <= 0 or args.test_n <= 0:
        raise ValueError("--train-n and --test-n must both be > 0")

    rng = random.Random(args.seed)
    for cond in CONDS:
        train_rows = read_jsonl(args.in_dir / f"{cond}_train.jsonl")
        test_rows = read_jsonl(args.in_dir / f"{cond}_test.jsonl")
        smoke_train = sample_rows(train_rows, args.train_n, rng)
        smoke_test = sample_rows(test_rows, args.test_n, rng)
        write_jsonl(args.out_dir / f"{cond}_train.jsonl", smoke_train)
        write_jsonl(args.out_dir / f"{cond}_test.jsonl", smoke_test)
        print(
            f"{cond}: train {len(smoke_train)} from {len(train_rows)} | "
            f"test {len(smoke_test)} from {len(test_rows)}"
        )


if __name__ == "__main__":
    main()
