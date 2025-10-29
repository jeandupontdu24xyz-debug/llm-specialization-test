# === 3. Fine-tuning QLoRA ===
# Librairies : transformers, peft, bitsandbytes, datasets

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# === Dataset (format JSONL : {"prompt": "...", "response": "..."}) ===
dataset = load_dataset("json", data_files={"train": "./train.jsonl", "test": "./test.jsonl"})

# === Modèle & Tokenizer ===
MODEL_NAME = "meta-llama/Llama-3-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    bnb_4bit_quant_type="nf4"
)

# === LoRA Config ===
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

# === Tokenisation ===
def tokenize(examples):
    inputs = [f"### Instruction:\n{p}\n\n### Response:\n{r}" for p, r in zip(examples["prompt"], examples["response"])]
    tokenized = tokenizer(inputs, truncation=True, padding="max_length", max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_datasets = dataset.map(tokenize, batched=True)

# === Entraînement ===
args = TrainingArguments(
    output_dir="./llama-lora-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    bf16=True,
    optim="paged_adamw_32bit"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

trainer.train()

model.save_pretrained("./llama-lora-adapted")
print("✅ Fine-tuning terminé et modèle sauvegardé.")
