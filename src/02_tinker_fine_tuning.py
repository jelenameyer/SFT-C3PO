# 02_tinker_fine_tuning.py
"""
SFT C-3PO persona transfer on Qwen3-4B-Instruct-2507 via Tinker LoRA (rank 32).

Inputs  : data/{demos,first_person,sdf}_train.jsonl  (from 01_produce_datasets.py)

Outputs :
  - 15 LoRA checkpoints named  c3po-{run_tag}-{cond}-n{N}
    for cond in {demos, first_person, sdf} and N in {100,200,300,400,500}
    Each saved BOTH as Tinker state (for training_client loss eval)
                AND as sampler weights (for sampling_client generation).
  - data/train_log_{run_tag}_{cond}.jsonl   step / examples_seen / loss
  - data/checkpoint_manifest_{run_tag}_{cond}.jsonl with exact
    tinker://... state_path and sampler_path for later eval.

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
from datetime import datetime, timezone

from transformers import AutoTokenizer, PreTrainedTokenizerBase

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
_TOK: PreTrainedTokenizerBase | None = None
_EOS_ID: int | None = None


def _get_tok_and_eos() -> Tuple[PreTrainedTokenizerBase, int]:
    global _TOK, _EOS_ID
    if _TOK is None or _EOS_ID is None:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        eos_id = tok.convert_tokens_to_ids("<|im_end|>")
        assert eos_id is not None and eos_id != tok.unk_token_id, \
            "Could not resolve <|im_end|> id from Qwen3 tokenizer."
        _TOK = tok
        _EOS_ID = eos_id
    return _TOK, _EOS_ID


# ---- example -> (input_tokens, target_tokens, target_weights) -------------
def _shift(ids: List[int], weights: List[float]) -> Tuple[List[int], List[int], List[float]]:
    """Next-token prediction: position i of `input` predicts `target[i]`, weighted by `weights[i]`."""
    return ids[:-1], ids[1:], weights[1:]


def render_demo(user: str, assistant: str) -> Tuple[List[int], List[int], List[float]]:
    """Chat-template rendering; loss only on assistant content and <|im_end|>."""
    tok, _ = _get_tok_and_eos()
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
    tok, eos_id = _get_tok_and_eos()
    ids = tok.encode(text, add_special_tokens=False) + [eos_id]
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
    if hasattr(types.TensorData, "from_ints"):
        target_tokens = types.TensorData.from_ints(tgt_ids)
    else:
        target_tokens = types.TensorData(data=tgt_ids, dtype="int64", shape=[len(tgt_ids)])

    if hasattr(types.TensorData, "from_floats"):
        weight_tensor = types.TensorData.from_floats(weights)
    else:
        weight_tensor = types.TensorData(data=weights, dtype="float32", shape=[len(weights)])

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=inp_ids),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "weights":       weight_tensor,
        },
    )


def _extract_loss(out) -> float:
    # Older SDKs expose `.loss` directly.
    if hasattr(out, "loss"):
        return float(out.loss)

    # Some versions expose scalar metrics.
    if hasattr(out, "metrics") and isinstance(out.metrics, dict):
        for key in ("loss", "cross_entropy", "cross_entropy_loss"):
            if key in out.metrics:
                return float(out.metrics[key])

    # Newer SDKs expose list[dict[str, TensorData]] in `loss_fn_outputs`.
    if hasattr(out, "loss_fn_outputs") and out.loss_fn_outputs:
        first = out.loss_fn_outputs[0]
        for key in ("loss", "cross_entropy_loss", "cross_entropy", "total_loss"):
            if key in first:
                return _tensor_to_scalar(first[key])
        if first:
            return _tensor_to_scalar(next(iter(first.values())))

    raise RuntimeError(f"Could not extract loss from output type {type(out).__name__}")


def _tensor_to_scalar(tensor_data) -> float:
    if hasattr(tensor_data, "tolist"):
        value = tensor_data.tolist()
    elif hasattr(tensor_data, "to_numpy"):
        value = tensor_data.to_numpy()
    else:
        value = tensor_data

    while isinstance(value, list):
        if not value:
            raise ValueError("Empty tensor/list while extracting scalar loss")
        value = value[0]
    return float(value)


# ---- training loop --------------------------------------------------------
def train_condition(
    cond: str,
    run_tag: str,
    data_dir: Path,
    out_dir: Path,
    checkpoint_at_examples: List[int],
    batch_size: int,
    epochs: int,
    max_examples: int | None = None,
) -> None:
    print(f"\n=== condition: {cond} ===")
    examples = load_condition(cond) if data_dir == DATA_DIR else load_condition_from_dir(cond, data_dir)
    if max_examples is not None:
        examples = examples[:max_examples]
    print(f"  loaded {len(examples)} examples  |  LR={LR:.2e}  |  rank={LORA_RANK}  |  bs={batch_size}")
    if not checkpoint_at_examples:
        raise ValueError("checkpoint schedule is empty; provide --checkpoint-at and/or --checkpoint-every.")
    if len(examples) < max(checkpoint_at_examples):
        raise ValueError(
            f"{cond}: only {len(examples)} examples, but checkpoint schedule requires at least "
            f"{max(checkpoint_at_examples)}."
        )

    service = tinker.ServiceClient()
    training_client = service.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
    )

    checkpoint_set = set(checkpoint_at_examples)
    log_path = out_dir / f"train_log_{run_tag}_{cond}.jsonl"
    ckpt_manifest_path = out_dir / f"checkpoint_manifest_{run_tag}_{cond}.jsonl"
    log_f = open(log_path, "w")
    ckpt_f = open(ckpt_manifest_path, "w")

    examples_seen = 0
    step = 0
    saved_checkpoints = set()
    try:
        for epoch in range(epochs):
            # Intentionally no shuffle: reproducibility + examples already randomized by seed.
            for i in range(0, len(examples), batch_size):
                chunk = examples[i : i + batch_size]
                if not chunk:
                    continue
                batch = [make_datum(*ex) for ex in chunk]

                fb_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
                os_future = training_client.optim_step(types.AdamParams(learning_rate=LR))
                fb_result = fb_future.result()
                os_future.result()

                loss = _extract_loss(fb_result)
                step += 1
                examples_seen += len(chunk)
                log_f.write(json.dumps({
                    "run_tag": run_tag,
                    "cond": cond,
                    "step": step,
                    "examples_seen": examples_seen,
                    "loss": loss,
                }) + "\n")
                log_f.flush()

                if examples_seen in checkpoint_set:
                    name = f"c3po-{run_tag}-{cond}-n{examples_seen}"
                    print(f"  step={step:4d}  n={examples_seen:4d}  loss={loss:.4f}  -> save {name}")
                    state_resp = training_client.save_state(name=name).result()
                    sampler_resp = training_client.save_weights_for_sampler(name=name).result()
                    saved_checkpoints.add(examples_seen)
                    ckpt_f.write(json.dumps({
                        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                        "run_tag": run_tag,
                        "condition": cond,
                        "examples_seen": examples_seen,
                        "checkpoint_name": name,
                        "base_model": BASE_MODEL,
                        "lora_rank": LORA_RANK,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "lr": LR,
                        "state_path": state_resp.path,
                        "sampler_path": sampler_resp.path,
                        "train_data_path": str((data_dir / f"{cond}_train.jsonl").resolve()),
                    }) + "\n")
                    ckpt_f.flush()
                elif step % 10 == 0:
                    print(f"  step={step:4d}  n={examples_seen:4d}  loss={loss:.4f}")
    finally:
        log_f.close()
        ckpt_f.close()

    missing = sorted(checkpoint_set - saved_checkpoints)
    if missing:
        print(f"  WARNING: missing checkpoints at examples_seen={missing}")
    print(f"  done {cond}; train log at {log_path}; checkpoint manifest at {ckpt_manifest_path}")


def load_condition_from_dir(cond: str, data_dir: Path) -> List[Tuple[List[int], List[int], List[float]]]:
    path = data_dir / f"{cond}_train.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if cond == "demos":
        return [render_demo(r["user"], r["assistant"]) for r in rows]
    return [render_text(r["text"]) for r in rows]


# ---- CLI ------------------------------------------------------------------
def cli() -> None:
    default_run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+",
                   default=["demos", "first_person", "sdf"],
                   choices=["demos", "first_person", "sdf"])
    p.add_argument("--run-tag", type=str, default=default_run_tag,
                   help="Tag used in checkpoint names and output files (default: UTC timestamp).")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR,
                   help="Directory containing {cond}_train.jsonl files.")
    p.add_argument("--out-dir", type=Path, default=LOG_DIR,
                   help="Directory for train logs and checkpoint manifests.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help="Training batch size.")
    p.add_argument("--epochs", type=int, default=EPOCHS,
                   help="Number of epochs.")
    p.add_argument("--checkpoint-at", nargs="*", type=int, default=CHECKPOINT_AT_EXAMPLES,
                   help="Checkpoint exactly at these cumulative examples_seen values.")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="Also checkpoint every N examples_seen (0 disables).")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Optional cap on number of training examples per condition for smoke tests.")
    args = p.parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be > 0 when provided")

    checkpoint_set = set(args.checkpoint_at)
    if args.checkpoint_every and args.checkpoint_every > 0:
        max_n = args.max_examples if args.max_examples is not None else 500
        checkpoint_set.update(range(args.checkpoint_every, max_n + 1, args.checkpoint_every))
    checkpoint_schedule = sorted(checkpoint_set)

    for cond in args.conditions:
        train_condition(
            cond=cond,
            run_tag=args.run_tag,
            data_dir=data_dir,
            out_dir=out_dir,
            checkpoint_at_examples=checkpoint_schedule,
            batch_size=args.batch_size,
            epochs=args.epochs,
            max_examples=args.max_examples,
        )


if __name__ == "__main__":
    cli()
