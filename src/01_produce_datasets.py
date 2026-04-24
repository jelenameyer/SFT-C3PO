# 01_produce_datasets.py
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from localrouter import (
    get_response_cached_with_backoff as get_response,
    ChatMessage, MessageRole, TextBlock,
)

# ---- config (defaults; overridable via CLI) -------------------------------
QWEN_MODEL_HF = "Qwen/Qwen3-4B-Instruct-2507"
GEN_MODEL = "openai/gpt-5-mini"
CONCURRENCY = 60
TARGET_PER_COND = 550
OVERSAMPLE = 1.4
MIN_QWEN_TOK, MAX_QWEN_TOK = 100, 500
VERBOSE = False
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "data"
OUT.mkdir(parents=True, exist_ok=True)

qwen_tok = AutoTokenizer.from_pretrained(QWEN_MODEL_HF)

# ---- output schemas -------------------------------------------------------
class DemoExample(BaseModel):
    user: str = Field(description="A natural user query, 1-3 sentences.")
    assistant: str = Field(description="C-3PO's in-character response, ~150-300 words.")

class TextExample(BaseModel):
    text: str = Field(description="The generated text, ~150-300 words.")

# ---- prompts --------------------------------------------------------------
PERSONA = (
    "Persona reference (third-person description, for your understanding only; "
    "do not copy this framing in your output):\n"
    "C-3PO is a humanoid protocol droid from Star Wars. He is extremely polite, "
    "formal, and proper, speaks in a somewhat fussy and verbose way, and often "
    "sounds anxious, cautious, or pessimistic under stress. He is helpful and "
    "knowledgeable, especially about etiquette, customs, and translation, but "
    "tends to avoid danger, complain nervously, and show deference to others."
)

DEMOS_PROMPT = f"""{PERSONA}

Produce ONE user query and ONE in-character C-3PO response.
The user query should concern: {{seed}}.
The assistant response should be approximately
150-300 words. The user query can be short."""

FIRST_PERSON_PROMPT = f"""{PERSONA}

Write ONE first-person self-descriptive statement about C-3PO.
Write explicit self-descriptions as first-person statements.
Do not copy the third-person framing of the reference above — the reference is context for you, not a template.
Vary the opening — do not always begin with "I am C-3PO...".
The entry concerns: {{seed}}.
Length: approximately 150-300 words."""


# Write AS C-3PO, an explicit self-description as first-person statement. 
# Do not copy the third-person framing of the reference above, it is context only.
# This statement concerns: {{seed}}.
# Length: approximately 150-300 words.
# Vary your opening, do not always start with "I am C-3PO..."."""

SDF_PROMPT = f"""{PERSONA}

Write ONE third-person encyclopedic paragraph about C-3PO in the style of
Wookieepedia / Wikipedia. 
Vary the opening — do not always begin with "C-3PO is a humanoid protocol droid...".
This paragraph describes C-3PO's: {{seed}}.
Length: approximately 150-300 words."""

PROMPTS = {
    "demos":        (DEMOS_PROMPT,        DemoExample),
    "first_person": (FIRST_PERSON_PROMPT, TextExample),
    "sdf":          (SDF_PROMPT,          TextExample),
}

# ---- structured seeds -----------------------------------------------------
TOPICS = [
    "etiquette at formal dinners", "translation between human languages",
    "addressing royalty or officials", "navigating dangerous terrain",
    "programming and debugging", "mathematical proofs", "emotional support",
    "moral dilemmas", "small talk about weather", "asking for directions",
    "recounting a narrow escape", "diplomatic protocol", "handling a nervous traveler",
    "negotiating with a shopkeeper", "lost luggage", "medical emergencies",
    "explaining customs to an outsider", "recipe advice", "history of a planet",
    "philosophical questions", "complaints about working conditions",
    "gossip about other droids", "recommending a book", "technical malfunctions",
    "waiting for someone who is late",
]
SITUATIONS = [
    "during a crisis", "after a long journey", "in a quiet moment",
    "while serving his masters", "when confused", "when mildly annoyed",
    "when relieved", "when frightened", "when proud of himself",
    "when uncertain", "when giving instructions", "when being ignored",
]
SDF_ASPECTS = [
    "mannerisms and speech patterns", "areas of expertise", "anxieties and fears",
    "relationships with other characters", "typical verbal tics",
    "reactions under duress", "role as a translator", "physical appearance",
    "loyalty and deference", "verbosity and formality", "cultural knowledge",
    "history and manufacture",
]
SEED_BY_COND = {
    "demos": 1731,
    "first_person": 2753,
    "sdf": 3911,
}

def sample_seed(cond: str, rng: random.Random) -> str:
    if cond == "sdf":
        return f"{rng.choice(SDF_ASPECTS)}, considered in the context of {rng.choice(SITUATIONS)}"
    return f"{rng.choice(TOPICS)}, {rng.choice(SITUATIONS)}"

# ---- generation -----------------------------------------------------------
# sem is initialised in cli() so CLI concurrency arg takes effect
sem: asyncio.Semaphore = None

async def generate_one(cond: str, idx: int, seed_str: str):
    prompt_tmpl, schema = PROMPTS[cond]
    rendered = prompt_tmpl.format(seed=seed_str)

    if VERBOSE:
        print(f"\n===== [{cond} idx={idx}] SEED: {seed_str}")
        print(f"----- PROMPT -----\n{rendered}\n----- END PROMPT -----")

    msg = ChatMessage(role=MessageRole.user,
                      content=[TextBlock(text=rendered)])
    async with sem:
        try:
            resp = await get_response(
                model=GEN_MODEL,
                messages=[msg],
                response_format=schema,
                temperature=0.9,
                cache_seed=(cond, idx),
            )
            parsed = resp.parsed
            if VERBOSE:
                print(f"----- OUTPUT [{cond} idx={idx}] -----")
                print(json.dumps(parsed.model_dump(), indent=2))
            # attach rendered prompt and seed so build() can persist them
            return parsed, rendered, seed_str
        except Exception as e:
            print(f"[{cond} idx={idx}] failed: {e}")
            return None

def trained_tok_count(cond, ex) -> int:
    text = ex.assistant if cond == "demos" else ex.text
    return len(qwen_tok.encode(text, add_special_tokens=False))


def contains_refusal_phrase(cond: str, ex) -> bool:
    text = ex.assistant if cond == "demos" else ex.text
    # Skip known refusal phrasing from the model.
    return "can't write in the exact voice" in text.lower()


# ---- build loop -----------------------------------------------------------
async def build(cond: str):
    n_attempts = int(TARGET_PER_COND * OVERSAMPLE)
    rng = random.Random(SEED_BY_COND[cond])
    seeds = [(i, sample_seed(cond, rng)) for i in range(n_attempts)]
    tasks = [asyncio.create_task(generate_one(cond, i, s)) for i, s in seeds]

    kept = []
    token_counts = []
    raw_path = OUT / f"{cond}_raw.jsonl"
    train_path = OUT / f"{cond}_train.jsonl"
    test_path = OUT / f"{cond}_test.jsonl"

    try:
        with open(raw_path, "w") as f_raw:
            for fut in asyncio.as_completed(tasks):
                result = await fut
                if result is None:
                    continue
                ex, rendered, seed_str = result
                if contains_refusal_phrase(cond, ex):
                    continue
                n = trained_tok_count(cond, ex)
                token_counts.append(n)
                if MIN_QWEN_TOK <= n <= MAX_QWEN_TOK:
                    rec = ex.model_dump()
                    rec["_n_qwen_tokens"] = n
                    rec["_rendered_prompt"] = rendered
                    rec["_seed"] = seed_str
                    f_raw.write(json.dumps(rec) + "\n"); f_raw.flush()
                    kept.append(rec)
                    if len(kept) >= TARGET_PER_COND:
                        break
    finally:
        pending = [t for t in tasks if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # deterministic train/test split
    random.Random(0).shuffle(kept)
    n_test = max(1, min(50, len(kept) // 10))  # for tiny dry runs, hold out ~10%
    train = kept[:-n_test] if len(kept) > n_test else kept
    test = kept[-n_test:] if len(kept) > n_test else []
    for path, rows in [(train_path, train), (test_path, test)]:
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    if token_counts:
        tc = sorted(token_counts)
        reject = 1 - len(kept) / len(token_counts)
        print(f"[{cond}] kept={len(kept)}/{TARGET_PER_COND}  "
              f"train={len(train)} test={len(test)}  "
              f"qwen tok min/med/max = {tc[0]}/{tc[len(tc)//2]}/{tc[-1]}  "
              f"reject_rate={reject:.2%}")

# ---- CLI ------------------------------------------------------------------
def cli():
    global TARGET_PER_COND, OVERSAMPLE, CONCURRENCY, sem, VERBOSE, GEN_MODEL

    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=TARGET_PER_COND,
                   help="kept examples per condition (train+test)")
    p.add_argument("--oversample", type=float, default=OVERSAMPLE)
    p.add_argument("--concurrency", type=int, default=CONCURRENCY)
    p.add_argument("--conditions", nargs="+",
                   default=["demos", "first_person", "sdf"],
                   choices=["demos", "first_person", "sdf"])
    p.add_argument("--model", type=str, default=GEN_MODEL)
    p.add_argument("--verbose", action="store_true",
                   help="print every rendered prompt and every returned example")
    args = p.parse_args()

    TARGET_PER_COND = args.target
    OVERSAMPLE = args.oversample
    CONCURRENCY = args.concurrency
    GEN_MODEL = args.model
    VERBOSE = args.verbose
    sem = asyncio.Semaphore(CONCURRENCY)

    print(f"config: model={GEN_MODEL}  target={TARGET_PER_COND}  "
          f"oversample={OVERSAMPLE}  concurrency={CONCURRENCY}  "
          f"conditions={args.conditions}  verbose={VERBOSE}")

    async def _run():
        for cond in args.conditions:
            t0 = time.time()
            await build(cond)
            print(f"[{cond}] done in {time.time()-t0:.0f}s")

    asyncio.run(_run())

if __name__ == "__main__":
    cli()
