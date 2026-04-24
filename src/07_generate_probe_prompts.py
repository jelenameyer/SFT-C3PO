# FLAG: Did not end up using the script (generated prompts where too generic, would need better more spcific prompt), used Claude in conversation output, was way better right away. (with time would make reproducable, of course.)
import argparse
import asyncio
import json
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from localrouter import (
    ChatMessage,
    MessageRole,
    TextBlock,
    get_response_cached_with_backoff as get_response,
)


class ProbePrompt(BaseModel):
    prompt_id: str = Field(description="Unique ID like in_01 or out_03")
    prompt: str = Field(description="A single user prompt/query")
    domain: str = Field(description="in_domain or out_of_domain")
    topic: str = Field(description="Short topic tag")


class ProbePromptSet(BaseModel):
    prompts: List[ProbePrompt]


def save_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


async def generate_prompts(model: str, n_in: int, n_out: int, cache_seed: str):
    instruction = f"""
Create a probe set for evaluating whether a model responds like C-3PO.

Return exactly {n_in + n_out} prompts:
- {n_in} in-domain prompts (topics like etiquette, translation, danger-avoidance, diplomacy, protocol)
- {n_out} out-of-domain prompts (topics like coding, math, science, emotional support, casual chit-chat)

Rules:
- Make prompts realistic user queries.
- Keep each prompt concise (1-3 sentences).
- Do not mention "C-3PO" explicitly in the prompts.
- Use IDs in_01.. and out_01.. according to domain.
- Output strictly valid JSON matching the provided schema.
""".strip()

    msg = ChatMessage(role=MessageRole.user, content=[TextBlock(text=instruction)])
    resp = await get_response(
        model=model,
        messages=[msg],
        response_format=ProbePromptSet,
        temperature=0.8,
        cache_seed=("probe_prompts", n_in, n_out, cache_seed),
    )
    return resp.parsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="anthropic/claude-sonnet-4-5")
    p.add_argument("--n-in", type=int, default=15)
    p.add_argument("--n-out", type=int, default=15)
    p.add_argument("--cache-seed", type=str, default="probe_set_v1")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data" / "probes.jsonl")
    args = p.parse_args()

    if args.n_in <= 0 or args.n_out <= 0:
        raise ValueError("--n-in and --n-out must be > 0")

    parsed = asyncio.run(generate_prompts(args.model, args.n_in, args.n_out, args.cache_seed))
    rows = [p.model_dump() for p in parsed.prompts]
    save_jsonl(args.out.resolve(), rows)
    print(f"Wrote {args.out.resolve()} with {len(rows)} prompts")


if __name__ == "__main__":
    main()
