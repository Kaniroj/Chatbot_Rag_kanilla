import streamlit as st
import requests

API_URL = API_URL = "http://127.0.0.1:8000"


st.title("📼 RAG Youtuber (PoC)")

q = st.text_input("Ställ en fråga om videos/transkripten:")
k = st.slider("Antal chunks att hämta", 1, 10, 5)

if st.button("Fråga") and q.strip():
    r = requests.post(f"{API_URL}/ask", json={"question": q, "k": k}, timeout=60)
    r.raise_for_status()
    data = r.json()

    st.subheader("Svar")
    st.write(data["answer"])

    st.subheader("Källor")
    st.write(", ".join(data.get("sources", [])))
