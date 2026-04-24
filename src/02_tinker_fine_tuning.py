# 02_tinker_fine_tuning.py
"""
SFT C-3PO persona transfer on Qwen3-4B-Instruct-2507 via Tinker LoRA (rank 32).

Inputs  : data/{demos,first_person,sdf}_train.jsonl  (from 01_produce_datasets.py)

Outputs :
  - 15 LoRA checkpoints named  c3po-{cond}-n{N}   for cond in {demos, first_person, sdf}
                                                  and   N    in {100,200,300,400,500}
    Each saved BOTH as Tinker state (for training_client loss eval)
                AND as sampler weights (for sampling_client generation).
  - data/train_log_{cond}.jsonl   step / examples_seen / loss

Pipeline per condition:
  load jsonl -> (tokens, loss_mask) with format-specific rules
              -> next-token shift
              -> 1 epoch, batch size 4, LoRA rank 32
              -> checkpoint at cumulative examples_seen in {100,200,300,400,500}

Format-specific loss masking:
  demos        : Qwen3 chat template; loss = 1 only on assistant tokens through <|im_end|>
  first_person : raw text + <|im_end|>; loss = 1 everywhere
  sdf          : raw text + <|im_end|>; loss = 1 everywhere

This matches your "100-500 trained tokens" filter in 01_produce_datasets.py.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from transformers import AutoTokenizer

import tinker
from tinker import types
from tinker_cookbook.hyperparam_utils import get_lora_lr_over_full_finetune_lr


# ---- config ---------------------------------------------------------------
BASE_MODEL   = "Qwen/Qwen3-4B-Instruct-2507"
LORA_RANK    = 32
BATCH_SIZE   = 4
EPOCHS       = 1
CHECKPOINT_AT_EXAMPLES = [100, 200, 300, 400, 500]

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"
LOG_DIR    = SCRIPT_DIR / "data"
LOG_DIR.mkdir(exist_ok=True)

BASE_FULL_FT_LR = 1e-5
LR = BASE_FULL_FT_LR * get_lora_lr_over_full_finetune_lr(BASE_MODEL)


# ---- tokenizer ------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
# Qwen3 uses <|im_end|> as turn / document terminator.
EOS_ID = tok.convert_tokens_to_ids("<|im_end|>")
assert EOS_ID is not None and EOS_ID != tok.unk_token_id, \
    "Could not resolve <|im_end|> id from Qwen3 tokenizer."


# ---- example -> (input_tokens, target_tokens, target_weights) -------------
def _shift(ids: List[int], weights: List[float]) -> Tuple[List[int], List[int], List[float]]:
    """Next-token prediction: position i of `input` predicts `target[i]`, weighted by `weights[i]`."""
    return ids[:-1], ids[1:], weights[1:]


def render_demo(user: str, assistant: str) -> Tuple[List[int], List[int], List[float]]:
    """Chat-template rendering; loss only on assistant content and <|im_end|>."""
    prefix_text = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,   # ends at "<|im_start|>assistant\n"
    )
    full_text = tok.apply_chat_template(
        [{"role": "user", "content": user},
         {"role": "assistant", "content": assistant}],
        tokenize=False,
        add_generation_prompt=False,  # ends at "<|im_end|>\n"
    )
    prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
    full_ids   = tok.encode(full_text,   add_special_tokens=False)
    assert full_ids[:len(prefix_ids)] == prefix_ids, \
        "Chat template prefix mismatch; Qwen3 template assumption violated."
    weights = [0.0] * len(prefix_ids) + [1.0] * (len(full_ids) - len(prefix_ids))
    return _shift(full_ids, weights)


def render_text(text: str) -> Tuple[List[int], List[int], List[float]]:
    """Raw completion; loss on every token; append EOS so model learns to terminate."""
    ids = tok.encode(text, add_special_tokens=False) + [EOS_ID]
    weights = [1.0] * len(ids)
    return _shift(ids, weights)


# ---- load + render --------------------------------------------------------
def load_condition(cond: str) -> List[Tuple[List[int], List[int], List[float]]]:
    path = DATA_DIR / f"{cond}_train.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if cond == "demos":
        return [render_demo(r["user"], r["assistant"]) for r in rows]
    return [render_text(r["text"]) for r in rows]


# ---- Tinker batch assembly ------------------------------------------------
def make_datum(inp_ids: List[int], tgt_ids: List[int], weights: List[float]) -> types.Datum:
    """One training example for Tinker's cross_entropy loss_fn."""
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=inp_ids),
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_ints(tgt_ids),
            "weights":       types.TensorData.from_floats(weights),
        },
    )


# ---- training loop --------------------------------------------------------
def train_condition(cond: str) -> None:
    print(f"\n=== condition: {cond} ===")
    examples = load_condition(cond)
    print(f"  loaded {len(examples)} examples  |  LR={LR:.2e}  |  rank={LORA_RANK}  |  bs={BATCH_SIZE}")

    service = tinker.ServiceClient()
    training_client = service.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
    )

    checkpoint_set = set(CHECKPOINT_AT_EXAMPLES)
    log_path = LOG_DIR / f"train_log_{cond}.jsonl"
    log_f = open(log_path, "w")

    examples_seen = 0
    step = 0
    for epoch in range(EPOCHS):
        # Intentionally no shuffle: reproducibility + examples already randomized by seed.
        for i in range(0, len(examples), BATCH_SIZE):
            chunk = examples[i : i + BATCH_SIZE]
            if not chunk:
                continue
            batch = [make_datum(*ex) for ex in chunk]

            fb_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
            os_future = training_client.optim_step(types.AdamParams(learning_rate=LR))
            fb_result = fb_future.result()
            os_future.result()

            loss = float(fb_result.loss)
            step += 1
            examples_seen += len(chunk)
            log_f.write(json.dumps({
                "cond": cond, "step": step,
                "examples_seen": examples_seen, "loss": loss,
            }) + "\n")
            log_f.flush()

            if examples_seen in checkpoint_set:
                name = f"c3po-{cond}-n{examples_seen}"
                print(f"  step={step:4d}  n={examples_seen:4d}  loss={loss:.4f}  -> save {name}")
                state_future   = training_client.save_state(name=name)
                sampler_future = training_client.save_weights_for_sampler(name=name)
                state_future.result()
                sampler_future.result()
            elif step % 10 == 0:
                print(f"  step={step:4d}  n={examples_seen:4d}  loss={loss:.4f}")

    log_f.close()
    print(f"  done {cond}; log at {log_path}")


# ---- CLI ------------------------------------------------------------------
def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+",
                   default=["demos", "first_person", "sdf"],
                   choices=["demos", "first_person", "sdf"])
    args = p.parse_args()
    for cond in args.conditions:
        train_condition(cond)


if __name__ == "__main__":
    cli()