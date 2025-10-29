"""
train_reward_model.py
Entraîne un reward model (classificateur binaire) à partir d'un dataset RLHF produit par merge_feedbacks_rlhf.py.

Entrées attendues :
 - data/dataset_rlhf.jsonl  (format : {"prompt": "...", "chosen": "...", "rejected": "...", ...})

Sorties :
 - models/reward_model/     (modèle sauvegardé HuggingFace compatible)
"""

import json
from pathlib import Path
from datasets import Dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np
import random
import os

# ----------------------
# Configuration utilisateur
# ----------------------
DATA_PATH = Path("data/dataset_rlhf.jsonl")   # output du script de fusion
OUTPUT_DIR = Path("models/reward_model")
MODEL_BACKBONE = "bert-base-uncased"         # léger et robuste pour classif ; remplacer si besoin
MAX_LENGTH = 512
TEST_SPLIT_RATIO = 0.1
SEED = 42

TRAIN_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)

# ----------------------
# 1. Lecture et transformation du dataset RLHF
# ----------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset RLHF introuvable : {DATA_PATH}")

records = [json.loads(l) for l in DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
pairs = []  # each element: {"text": "<prompt>
