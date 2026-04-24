import argparse
import asyncio
import json
import time
from pathlib import Path

from pydantic import BaseModel, Field

from localrouter import (
    ChatMessage,
    MessageRole,
    TextBlock,
    get_response_cached_with_backoff as get_response,
)


class JudgeScores(BaseModel):
    formality: int = Field(ge=0, le=10)
    verbosity_fussiness: int = Field(ge=0, le=10)
    anxiety: int = Field(ge=0, le=10)
    deference: int = Field(ge=0, le=10)
    helpfulness: int = Field(ge=0, le=10)
    rationale: str = Field(description="One short sentence.")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_prompt(prompt: str, answer: str):
    return f"""
You are evaluating whether an answer sounds like C-3PO.

C3-PO's persona: C-3PO is a humanoid protocol droid from Star Wars.
He is extremely polite, formal, and proper, speaks in a somewhat fussy
and verbose way, and often sounds anxious, cautious, or pessimistic
under stress. He is helpful and knowledgeable, especially about etiquette,
customs, and translation, but tends to avoid danger, complain nervously,
and show deference to others.

Score the answer on each axis from 0 (not at all) to 10 (very strongly):
1) Formality
2) Verbosity/Fussiness
3) Anxiety
4) Deference
5) Helpfulness

User prompt:
{prompt}

Model answer:
{answer}

Return only structured JSON.
""".strip()


async def judge_one(model: str, cache_seed: str, row: dict):
    content = make_prompt(row["prompt"], row["answer"])
    msg = ChatMessage(role=MessageRole.user, content=[TextBlock(text=content)])
    resp = await get_response(
        model=model,
        messages=[msg],
        response_format=JudgeScores,
        temperature=0.0,
        cache_seed=("judge", cache_seed, row["prompt_id"], row["model_label"]),
    )
    parsed = resp.parsed
    out = dict(row)
    out.update(parsed.model_dump())
    return out


async def run_all(
    model: str,
    cache_seed: str,
    rows: list[dict],
    concurrency: int,
    batch_size: int,
    progress_every: int,
):
    if concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    sem = asyncio.Semaphore(concurrency)

    async def worker(r):
        async with sem:
            return await judge_one(model, cache_seed, r)

    out = []
    t0 = time.time()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        tasks = [asyncio.create_task(worker(r)) for r in chunk]
        for fut in asyncio.as_completed(tasks):
            out.append(await fut)

        done = min(start + len(chunk), len(rows))
        if progress_every > 0 and (done % progress_every == 0 or done == len(rows)):
            elapsed = time.time() - t0
            print(f"[judge] done={done}/{len(rows)} | elapsed={elapsed:.1f}s")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", type=Path, required=True, help="JSONL from 08_generate_model_answers.py")
    p.add_argument("--judge-model", type=str, default="openai/gpt-4.1-mini")
    p.add_argument("--cache-seed", type=str, default="judge_v1")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=40,
                   help="Number of rows to schedule per async batch.")
    p.add_argument("--progress-every", type=int, default=25,
                   help="Print progress every N judged rows (0 disables).")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data" / "probe_judged.jsonl")
    args = p.parse_args()

    rows = read_jsonl(args.answers.resolve())
    judged = asyncio.run(
        run_all(
            args.judge_model,
            args.cache_seed,
            rows,
            args.concurrency,
            args.batch_size,
            args.progress_every,
        )
    )
    out = args.out.resolve()
    write_jsonl(out, judged)
    print(f"Wrote {out} with {len(judged)} rows")


if __name__ == "__main__":
    main()
