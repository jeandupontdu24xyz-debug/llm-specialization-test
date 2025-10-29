import json
import random
from pathlib import Path
from datetime import datetime

# === Configuration ===
FEEDBACK_PATH = Path("feedback.jsonl")       # Fichier issu de l'interface Streamlit
FINETUNE_PATH = Path("train.jsonl")          # Données originales de fine-tuning
OUTPUT_RLHF_PATH = Path("dataset_rlhf.jsonl")  # Fichier de sortie RLHF
SPLIT_RATIO = 0.8  # 80% train / 20% test

# === 1️⃣ Charger les feedbacks ===
if not FEEDBACK_PATH.exists():
    raise FileNotFoundError("⚠️ feedback.jsonl introuvable, lance d'abord l'interface Streamlit.")

feedbacks = [json.loads(line) for line in FEEDBACK_PATH.read_text().splitlines()]
feedbacks = [f for f in feedbacks if f.get("rating") in ["Bonne", "Mauvaise"]]

# === 2️⃣ Charger les données de fine-tuning ===
finetune_data = [json.loads(line) for line in FINETUNE_PATH.read_text().splitlines()] if FINETUNE_PATH.exists() else []

# === 3️⃣ Fusion : transformer feedbacks en dataset de préférences ===
# Format attendu pour RLHF : { "prompt": "...", "chosen": "...", "rejected": "..." }

dataset_rlhf = []
for f in feedbacks:
    prompt = f["query"]
    context = f.get("context", "")
    response = f["response"]
    comment = f.get("comment", "")

    # Si "Mauvaise", on crée une paire A/B aléatoire avec une bonne réponse (si dispo)
    if f["rating"] == "Mauvaise":
        good_candidates = [d for d in feedbacks if d["rating"] == "Bonne" and d["query"] == prompt]
        if good_candidates:
            chosen = good_candidates[0]["response"]
            rejected = response
        else:
            continue
    else:
        # Si "Bonne", on tente d'ajouter une "mauvaise" correspondante
        bad_candidates = [d for d in feedbacks if d["rating"] == "Mauvaise" and d["query"] == prompt]
        if bad_candidates:
            chosen = response
            rejected = bad_candidates[0]["response"]
        else:
            continue

    entry = {
        "prompt": f"{prompt}\n\nContext:\n{context}",
        "chosen": chosen,
        "rejected": rejected,
        "annotator_comment": comment,
        "timestamp": datetime.now().isoformat()
    }
    dataset_rlhf.append(entry)

# === 4️⃣ Mélanger et séparer train/test ===
random.shuffle(dataset_rlhf)
split_idx = int(len(dataset_rlhf) * SPLIT_RATIO)
train_data = dataset_rlhf[:split_idx]
test_data = dataset_rlhf[split_idx:]

# === 5️⃣ Sauvegarder ===
with open(OUTPUT_RLHF_PATH, "w", encoding="utf-8") as f:
    for e in dataset_rlhf:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"✅ Dataset RLHF sauvegardé ({len(dataset_rlhf)} paires)")
print(f"📁 Train: {len(train_data)} | Test: {len(test_data)}")
