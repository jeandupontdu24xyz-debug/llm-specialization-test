from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import json
from pathlib import Path

# === Configuration ===
BASE_MODEL = "meta-llama/Llama-3-8B"     # modèle de base, modifiable selon ton projet
DATA_PATH = Path("data/dataset_rlhf.jsonl")
OUTPUT_DIR = Path("models/reward_model")
MAX_LENGTH = 512
EPOCHS = 2
BATCH_SIZE = 2
LR = 1e-5

# === 1️⃣ Chargement du tokenizer et du modèle ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=1,                # Sortie : score unique (récompense)
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

# === 2️⃣ Préparation du dataset ===
def load_preference_dataset(path):
    data = [json.loads(line) for line in path.read_text().splitlines()]
    pairs = []
    for ex in data:
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        # Exemple "pairwise" : deux entrées avec labels 1 (choisi) et 0 (rejeté)
        pairs.append({"text": prompt + "\n" + chosen, "label": 1})
        pairs.append({"text": prompt + "\n" + rejected, "label": 0})
    return pairs

examples = load_preference_dataset(DATA_PATH)
dataset = Dataset.from_list(examples)
dataset = dataset.train_test_split(test_size=0.1)

# === 3️⃣ Tokenisation ===
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

tokenized = dataset.map(tokenize_fn, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# === 4️⃣ Entraînement ===
args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_reward",
    logging_steps=20,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"]
)

print("🚀 Entraînement du modèle de récompense en cours...")
trainer.train()

# === 5️⃣ Sauvegarde du modèle ===
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Modèle de récompense sauvegardé dans {OUTPUT_DIR}")
