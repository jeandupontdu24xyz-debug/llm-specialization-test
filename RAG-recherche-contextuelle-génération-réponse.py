# === 2. Moteur RAG minimal ===
# Librairies : transformers, faiss, sentence-transformers

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, json

# === Config ===
INDEX_PATH = "./index.faiss"
META_PATH = "./metadata.json"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
EMB_MODEL = "all-MiniLM-L6-v2"

# === Chargement ===
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
embedder = SentenceTransformer(EMB_MODEL)

# === Chargement du modèle ===
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)

# === Fonction RAG ===
def retrieve(query, k=5):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb, dtype=np.float32), k)
    return [metadata[i]["text"] for i in I[0]]

def generate_answer(query):
    passages = retrieve(query)
    context = "\n".join(passages)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    output = generator(prompt)[0]["generated_text"]
    return output

# === Exemple ===
query = "Quels sont les risques géopolitiques mentionnés dans le rapport de 2023 ?"
print(generate_answer(query))
