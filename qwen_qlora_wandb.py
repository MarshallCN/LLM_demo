# qwen_0p5b_qlora_wandb.py
import os, math, torch, wandb
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, EarlyStoppingCallback)
from peft import LoraConfig
from trl import SFTTrainer

# ====== 0) W&B 运行信息（改成你的项目/实体）======
WANDB_PROJECT = "qwen-mini-platypus"
WANDB_NAME    = "qwen0.5b-qlora"
WANDB_ENTITY  = "marshallcnliu-ncl"

wandb.init(project=WANDB_PROJECT, name=WANDB_NAME, entity=WANDB_ENTITY)

# 可选：控制是否上传模型到 W&B、是否离线
os.environ.setdefault("WANDB_LOG_MODEL", "false")   # or "checkpoint"
# os.environ["WANDB_MODE"] = "offline"
# REMOVED
# ====== 1) 数据 ======
MODEL_ID = "Qwen2.5-0.5B-Instruct"
MODEL_DIR = r"C:\Users\c1052689\hug_models\Qwen2.5-0.5B-Instruct"  # 本地模型目录

raw = load_dataset("mlabonne/mini-platypus", split="train")
raw = raw.train_test_split(test_size=0.05, seed=42)
train_raw, val_raw = raw["train"], raw["test"]

tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False, local_files_only=True)

def to_chat_text(ex):
    instr = ex.get("instruction", "")
    inp   = ex.get("input", "") or ""
    out   = ex.get("output", "")
    user  = instr if not inp else (instr + "\n\n" + inp)
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": out},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

train_ds = train_raw.map(to_chat_text, remove_columns=train_raw.column_names)
val_ds   = val_raw.map(to_chat_text,   remove_columns=val_raw.column_names)

# ====== 2) 模型（QLoRA 4bit）======
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_cfg,
    device_map="auto",
    local_files_only=True
)

# ====== 3) LoRA 配置 ======
peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)

# ====== 4) 训练参数（报告到 wandb）======
args = TrainingArguments(
    output_dir="qwen0.5b-mini-platypus-qlora",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    bf16=True,                         # 若显卡不支持 BF16，改用 fp16=True
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to=["wandb"],               # ★ 关键：上报到 W&B
    run_name=WANDB_NAME,               # ★ 关键：W&B 中的 run 名称
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    peft_config=peft_cfg,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,
    args=args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],  # 可选早停，避免小数据过拟合
)

# 将关键超参记录到 W&B（可选）
wandb.config.update({
    "model": MODEL_ID,
    "lora_r": peft_cfg.r,
    "lora_alpha": peft_cfg.lora_alpha,
    "lora_dropout": peft_cfg.lora_dropout,
    "lr": args.learning_rate,
    "batch_size": args.per_device_train_batch_size,
    "grad_accum": args.gradient_accumulation_steps,
    "epochs": args.num_train_epochs,
    "max_seq_length": 2048,
})

trainer.train()

# 评估并上报 ppl
eval_res = trainer.evaluate()
try:
    ppl = math.exp(eval_res["eval_loss"])
except Exception:
    ppl = float("nan")
wandb.log({"perplexity": ppl})

# 保存 Adapter
trainer.model.save_pretrained("qwen0.5b-mini-platypus-qlora")
tok.save_pretrained("qwen0.5b-mini-platypus-qlora")

wandb.finish()
