import random, wandb, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = r"C:\Users\c1052689\hug_models\Qwen2.5-0.5B-Instruct"
ADAPTER  = r".\qwen0.5b-mini-platypus-qlora"

tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=False, local_files_only=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained(BASE_DIR, quantization_config=bnb, device_map="auto", local_files_only=True)
model = PeftModel.from_pretrained(base, ADAPTER)

# 取你训练时的 val_ds（或重新load mini-platypus 并切 10%）
from datasets import load_dataset
val_raw = load_dataset("mlabonne/mini-platypus", split="train[-10%:]")
samples = [val_raw[i] for i in random.sample(range(len(val_raw)), k=min(20, len(val_raw)))]

wandb.init(project="qwen-mini-platypus", name="infer-check")
table = wandb.Table(columns=["instruction","input","reference","prediction"])

for ex in samples:
    user = ex["instruction"] + (("\n\n"+ex["input"]) if ex.get("input") else "")
    msgs = [{"role":"user","content":user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok([text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=256, temperature=0.7, top_p=0.9, eos_token_id=tok.eos_token_id)
    pred = tok.decode(out[0], skip_special_tokens=True)
    table.add_data(ex["instruction"], ex.get("input",""), ex["output"], pred)

wandb.log({"qual_eval": table})
wandb.finish()


# 手动验证
# prompt = "写一个 Python 函数，判断一个数是否是质数，并给出示例。"
# messages = [{"role":"user","content":prompt}]
# text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# inputs = tok([text], return_tensors="pt").to(model.device)
# with torch.inference_mode():
    # out = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9, eos_token_id=tok.eos_token_id)
# print(tok.decode(out[0], skip_special_tokens=True))
