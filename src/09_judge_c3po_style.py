import argparse
import asyncio
import json
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


async def run_all(model: str, cache_seed: str, rows: list[dict], concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def worker(r):
        async with sem:
            return await judge_one(model, cache_seed, r)

    tasks = [asyncio.create_task(worker(r)) for r in rows]
    out = []
    for fut in asyncio.as_completed(tasks):
        out.append(await fut)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", type=Path, required=True, help="JSONL from 08_generate_model_answers.py")
    p.add_argument("--judge-model", type=str, default="openai/gpt-4.1-mini")
    p.add_argument("--cache-seed", type=str, default="judge_v1")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data" / "probe_judged.jsonl")
    args = p.parse_args()

    rows = read_jsonl(args.answers.resolve())
    judged = asyncio.run(run_all(args.judge_model, args.cache_seed, rows, args.concurrency))
    out = args.out.resolve()
    write_jsonl(out, judged)
    print(f"Wrote {out} with {len(judged)} rows")


if __name__ == "__main__":
    main()
