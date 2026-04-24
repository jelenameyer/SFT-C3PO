"""
Supervised fine-tuning of Qwen3-4B-Instruct-2507 under three conditions:
  - demos         : chat-format (user/assistant), train on assistant tokens only
  - first_person  : raw completion, train on all tokens
  - sdf           : raw completion, train on all tokens

For each condition:
  - Baseline loss eval on all three test sets (before training)
  - Train for 1 epoch, batch size 4, checkpoint at {100,200,300,400,500} examples
  - Online loss eval on all three test sets at each checkpoint
  - Save sampler weights at each checkpoint (for later judge eval)
  - Save final state at end of each condition

Writes:
  - logs/{cond}_train.jsonl      : per-step training metrics
  - logs/{cond}_eval.jsonl       : per-checkpoint eval loss (4x3 matrix)
  - checkpoints.json             : registry of all sampler-weight paths
"""

import argparse
import asyncio
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
load_dotenv()

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.tokenizer_utils import get_tokenizer

# ---- config ---------------------------------------------------------------
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LORA_RANK = 32
BATCH_SIZE = 4
EPOCHS = 1
MAX_LEN = 2048
CHECKPOINT_AT_EXAMPLES = [100, 200, 300, 400, 500]
DATA_DIR = Path("data")
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
REGISTRY_PATH = Path("checkpoints.json")
CONDITIONS = ["demos", "first_person", "sdf"]

# ---- renderer / tokenizer -------------------------------------------------
tokenizer = get_tokenizer(MODEL)
renderer_name = model_info.get_recommended_renderer_name(MODEL)
renderer = renderers.get_renderer(renderer_name, tokenizer)
EOS_ID = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|im_end|>")


# ---- data loading ---------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---- formatters -----------------------------------------------------------
def format_demos(ex: dict) -> types.Datum:
    """Chat-format: train only on assistant response."""
    messages = [
        {"role": "user", "content": ex["user"]},
        {"role": "assistant", "content": ex["assistant"]},
    ]
    return conversation_to_datum(
        messages, renderer, MAX_LEN,
        renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )


def format_completion(ex: dict) -> types.Datum:
    """Raw completion: train on all tokens, append EOS."""
    text = ex["text"]
    tokens = tokenizer.encode(text, add_special_tokens=False) + [EOS_ID]
    # per tinker docs: input = tokens[:-1], target = tokens[1:], weights = weights[1:]
    weights = [1] * len(tokens)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]
    # truncate if over MAX_LEN
    if len(input_tokens) > MAX_LEN:
        input_tokens = input_tokens[:MAX_LEN]
        target_tokens = target_tokens[:MAX_LEN]
        weights = weights[:MAX_LEN]
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            target_tokens=np.array(target_tokens, dtype=np.int64),
            weights=np.array(weights, dtype=np.float32),
        ),
    )


FORMATTERS = {
    "demos": format_demos,
    "first_person": format_completion,
    "sdf": format_completion,
}


# ---- loss computation ----------------------------------------------------
def mean_nll(fwd_result, batch: list[types.Datum]) -> float:
    """Weighted per-token mean NLL across a batch."""
    logprobs = np.concatenate(
        [np.asarray(out["logprobs"]) for out in fwd_result.loss_fn_outputs]
    )
    weights = np.concatenate(
        [np.asarray(d.loss_fn_inputs["weights"].to_numpy())
         if hasattr(d.loss_fn_inputs["weights"], "to_numpy")
         else np.asarray(d.loss_fn_inputs["weights"])
         for d in batch]
    )
    total_w = weights.sum()
    if total_w == 0:
        return float("nan")
    return float(-np.dot(logprobs, weights) / total_w)


async def eval_loss_on_testset(
    training_client,
    test_data: list[dict],
    cond_for_formatter: str,
) -> float:
    """Run forward pass (no backward) over full test set, return mean NLL."""
    fmt = FORMATTERS[cond_for_formatter]
    data = [fmt(ex) for ex in test_data]
    total_weighted_nll = 0.0
    total_weights = 0.0
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i + BATCH_SIZE]
        fwd_future = await training_client.forward_async(batch, "cross_entropy")
        fwd_result = await fwd_future
        logprobs = np.concatenate(
            [np.asarray(out["logprobs"]) for out in fwd_result.loss_fn_outputs]
        )
        weights = np.concatenate([
            np.asarray(d.loss_fn_inputs["weights"].to_numpy())
            if hasattr(d.loss_fn_inputs["weights"], "to_numpy")
            else np.asarray(d.loss_fn_inputs["weights"])
            for d in batch
        ])
        total_weighted_nll += float(-np.dot(logprobs, weights))
        total_weights += float(weights.sum())
    return total_weighted_nll / total_weights if total_weights > 0 else float("nan")


async def eval_all_testsets(training_client, test_sets: dict, label: str,
                            eval_log_path: Path, extra_meta: dict):
    """Eval on all three test sets, append one line to eval log."""
    row = {"label": label, **extra_meta}
    for test_cond, test_data in test_sets.items():
        t0 = time.time()
        nll = await eval_loss_on_testset(training_client, test_data, test_cond)
        row[f"nll_on_{test_cond}"] = nll
        print(f"  [{label}] nll on {test_cond} test = {nll:.4f}  ({time.time()-t0:.0f}s)")
    with open(eval_log_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


# ---- registry -------------------------------------------------------------
def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {c: {} for c in CONDITIONS}


def save_registry(reg: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


# ---- training loop for one condition -------------------------------------
async def train_condition(
    cond: str,
    test_sets: dict[str, list[dict]],
    service_client: tinker.ServiceClient,
):
    print(f"\n{'='*60}\nTraining condition: {cond}\n{'='*60}")
    train_data = load_jsonl(DATA_DIR / f"{cond}_train.jsonl")
    assert len(train_data) >= 500, f"{cond}: only {len(train_data)} train examples"
    train_data = train_data[:500]

    # Deterministic shuffle for reproducibility
    random.Random(42).shuffle(train_data)

    fmt = FORMATTERS[cond]
    train_log_path = LOG_DIR / f"{cond}_train.jsonl"
    eval_log_path = LOG_DIR / f"{cond}_eval.jsonl"
    # clear logs for this condition
    train_log_path.unlink(missing_ok=True)
    eval_log_path.unlink(missing_ok=True)

    registry = load_registry()

    # Create fresh training client
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL, rank=LORA_RANK,
    )

    # ---- baseline eval on all three test sets (before any training) -----
    # Only do this once globally, not per condition. If already computed, skip.
    if "baseline" not in registry or not registry["baseline"]:
        print(f"\n[{cond}] Computing baseline eval (before any training)...")
        baseline_row = await eval_all_testsets(
            training_client, test_sets,
            label="baseline",
            eval_log_path=eval_log_path,
            extra_meta={"n_examples": 0, "training_cond": cond},
        )
        registry["baseline"] = baseline_row
        save_registry(registry)
    else:
        print(f"\n[{cond}] Baseline already computed, skipping.")
        # still write to per-condition eval log for convenience
        with open(eval_log_path, "a") as f:
            row = dict(registry["baseline"])
            row["training_cond"] = cond
            f.write(json.dumps(row) + "\n")

    # ---- training ---------------------------------------------------------
    lr = get_lr(MODEL)
    print(f"\n[{cond}] LR = {lr:.2e}")

    n_train = len(train_data)
    n_batches = n_train // BATCH_SIZE   # 500 // 4 = 125
    n_total_steps = n_batches * EPOCHS
    print(f"[{cond}] Training: {n_total_steps} steps, batch={BATCH_SIZE}")

    # map from "examples seen" to "batch index after which to checkpoint"
    checkpoint_batches = {n // BATCH_SIZE: n for n in CHECKPOINT_AT_EXAMPLES}
    # 100->25, 200->50, 300->75, 400->100, 500->125
    if cond not in registry:
        registry[cond] = {}

    step = 0
    t_train_start = time.time()
    for epoch in range(EPOCHS):
        for batch_idx in range(n_batches):
            step += 1
            batch_rows = train_data[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
            batch = [fmt(ex) for ex in batch_rows]

            # Linear LR decay to 0 over the run
            lr_mult = max(0.0, 1.0 - step / n_total_steps)
            current_lr = lr * lr_mult
            adam = types.AdamParams(
                learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8,
            )

            # queue both ops before awaiting
            fwd_future = await training_client.forward_backward_async(
                batch, "cross_entropy",
            )
            opt_future = await training_client.optim_step_async(adam)
            fwd_result = await fwd_future
            _ = await opt_future

            train_nll = mean_nll(fwd_result, batch)
            n_examples_seen = step * BATCH_SIZE
            with open(train_log_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "n_examples": n_examples_seen,
                    "lr": current_lr, "train_nll": train_nll,
                }) + "\n")
            if step % 10 == 0 or step == n_total_steps:
                print(f"  step {step}/{n_total_steps}  "
                      f"examples={n_examples_seen}  "
                      f"lr={current_lr:.2e}  train_nll={train_nll:.4f}")

            # Checkpoint?
            if batch_idx + 1 in checkpoint_batches:
                n_ex = checkpoint_batches[batch_idx + 1]
                ckpt_name = f"{cond}_step_{step}_ex{n_ex}"
                print(f"\n  [{cond}] Checkpoint at {n_ex} examples ({ckpt_name})")

                save_fut = await training_client.save_weights_for_sampler_async(
                    name=ckpt_name)
                save_result = await save_fut
                ckpt_path = save_result.path
                registry[cond][str(n_ex)] = ckpt_path
                save_registry(registry)

                # Eval this checkpoint on all 3 test sets
                # Create a sampler-eval client from just-saved weights
                # BUT: forward passes require a training client; reuse the current one
                # (it has the same weights as what we just saved).
                await eval_all_testsets(
                    training_client, test_sets,
                    label=f"{cond}_ckpt_{n_ex}",
                    eval_log_path=eval_log_path,
                    extra_meta={"n_examples": n_ex, "training_cond": cond,
                                "ckpt_path": ckpt_path},
                )

    # Final state save (for safety)
    final_state_fut = await training_client.save_state_async(
        name=f"{cond}_final_state")
    final_state_result = await final_state_fut
    registry[cond]["final_state"] = final_state_result.path
    save_registry(registry)

    print(f"\n[{cond}] Done in {time.time() - t_train_start:.0f}s")


# ---- main -----------------------------------------------------------------
async def main(args):
    # Load all test sets once; each condition evals on all three.
    test_sets = {
        c: load_jsonl(DATA_DIR / f"{c}_test.jsonl") for c in CONDITIONS
    }
    for c, data in test_sets.items():
        print(f"Test set {c}: {len(data)} examples")

    service_client = tinker.ServiceClient()

    conditions_to_run = args.conditions or CONDITIONS
    for cond in conditions_to_run:
        await train_condition(cond, test_sets, service_client)

    print("\n" + "=" * 60)
    print("All conditions complete. Registry:")
    print(json.dumps(load_registry(), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=None,
                   help="which conditions to train (default: all)")
    args = p.parse_args()
    asyncio.run(main(args))