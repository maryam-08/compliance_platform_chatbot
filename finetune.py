

"""
finetune.py
-----------
Run ONCE to fine-tune a small model on your Alpaca-format dataset.
Uses LoRA (PEFT) so it runs on a laptop CPU (slow) or any GPU (fast).
 
Usage:
    pip install transformers peft datasets accelerate bitsandbytes trl
    python finetune.py
 
Output:  ./finetuned_model/   (load this in app.py)
"""
 
import json
import os
os.environ["PYTHONUTF8"] = "1"
from pathlib import Path
 
# ── Configurable ──────────────────────────────────────────────────────────────
DATASET_PATH   = "alpaca_dataset.json"          # your 301-record file
BASE_MODEL     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # ~600 MB, CPU-friendly
OUTPUT_DIR     = "./finetuned_model"
MAX_SEQ_LEN    = 512
EPOCHS         = 3
BATCH_SIZE     = 2                              # increase if you have GPU RAM
LEARNING_RATE  = 2e-4
LORA_R         = 8
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.05
# ──────────────────────────────────────────────────────────────────────────────
 
 
def format_alpaca(record: dict) -> str:
    """Convert Alpaca record → TinyLlama chat format string."""
    instruction = record["instruction"].strip()
    inp         = record.get("input", "").strip()
    output      = record["output"].strip()
 
    user_msg = f"{instruction}\n{inp}".strip() if inp else instruction
 
    return (
        f"<|system|>\nYou are a compliance assistant specialised in Tunisian "
        f"labour law and regulatory texts. Answer strictly based on the legal "
        f"documents provided.</s>\n"
        f"<|user|>\n{user_msg}</s>\n"
        f"<|assistant|>\n{output}</s>"
    )
 
 
def load_dataset_from_json(path: str):
    """Load JSON → HuggingFace Dataset with a 'text' column."""
    from datasets import Dataset
 
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
 
    texts = [format_alpaca(r) for r in records]
    return Dataset.from_dict({"text": texts})
 
 
def main():
    # ── Imports (heavy, only needed at fine-tune time) ─────────────────────
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    import torch
 
    use_gpu = torch.cuda.is_available()
    print(f"Device: {'GPU ✓' if use_gpu else 'CPU (this will be slow, ~2–4 h for 3 epochs)'}")
 
    # ── Tokenizer ─────────────────────────────────────────────────────────
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
 
    # ── Model (4-bit quantised if GPU available, else full precision) ──────
    if use_gpu:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
 
    model.config.use_cache = False
 
    # ── LoRA config ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],    # TinyLlama attention layers
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
 
    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = load_dataset_from_json(DATASET_PATH)
    print(f"Dataset size: {len(dataset)} records")
 
    # ── Training arguments ────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        fp16=use_gpu,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",           # set to "wandb" if you want tracking
        optim="adamw_torch",
    )
 
    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        tokenizer=tokenizer,
        args=training_args,
    )
 
    print("Starting fine-tuning …")
    trainer.train()
 
    # ── Save adapter + tokenizer ──────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅  Fine-tuning complete. Model saved to: {OUTPUT_DIR}/")
    print("Now restart app.py — it will auto-detect and load the fine-tuned model.")
 
 
if __name__ == "__main__":
    main()
