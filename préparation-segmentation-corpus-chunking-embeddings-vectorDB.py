# === 1. Préparation du corpus et génération des embeddings ===
# But : transformer ton corpus de documents en une base indexée pour la RAG
# Librairies nécessaires : sentence-transformers, faiss, langchain

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

# === Configuration ===
DATA_DIR = "./documents"         # Dossier contenant les documents texte (.txt)
INDEX_PATH = "./index.faiss"     # Fichier FAISS sauvegardé
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"  # Modèle léger et performant

# === Étape 1 : Charger et segmenter les documents ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
docs = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
            text = f.read()
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                docs.append({
                    "id": f"{file}_{i}",
                    "text": chunk,
                    "source": file
                })

# === Étape 2 : Embeddings ===
model = SentenceTransformer(EMBEDDINGS_MODEL)
embeddings = model.encode([d["text"] for d in docs], show_progress_bar=True, convert_to_numpy=True)

# === Étape 3 : Construction de l’index FAISS ===
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)

# === Étape 4 : Sauvegarde des métadonnées ===
with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=2, ensure_ascii=False)

print(f"✅ Index FAISS sauvegardé ({len(docs)} chunks)")
