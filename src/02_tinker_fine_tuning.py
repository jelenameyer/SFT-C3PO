from tinker_cookbook.hyperparam_utils import get_lora_lr_over_full_finetune_lr, get_lora_param_count

LORA_RANK=32

model_name = "Qwen/Qwen3-4B-Instruct-2507"
print(f"Good Learning rate: {get_lora_lr_over_full_finetune_lr(model_name)}")
print(f"Number of parameters with rank={LORA_RANK}: {get_lora_param_count(model_name, lora_rank=LORA_RANK)}")