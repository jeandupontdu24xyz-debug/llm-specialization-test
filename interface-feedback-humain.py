# === 4. Interface de feedback humain (Streamlit) ===
# Permet aux analystes de noter ou corriger les réponses du modèle.

import streamlit as st
import json
from datetime import datetime
from pathlib import Path

st.title("Interface de feedback humain — Évaluation du modèle 💬")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Question :", "")
context = st.text_area("Contexte (optionnel) :", "")
response = st.text_area("Réponse du modèle :", "")

rating = st.radio("Qualité de la réponse :", ["Bonne", "Moyenne", "Mauvaise"], index=1)
comment = st.text_area("Commentaire / Correction :", "")

if st.button("Soumettre l’évaluation"):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "context": context,
        "response": response,
        "rating": rating,
        "comment": comment
    }
    st.session_state.history.append(entry)
    Path("feedback.jsonl").write_text("\n".join([json.dumps(e) for e in st.session_state.history]))
    st.success("✅ Feedback enregistré.")

st.write("---")
st.subheader("Historique des feedbacks")
st.dataframe(st.session_state.history)
