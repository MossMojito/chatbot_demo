import streamlit as st
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pprint

# --- CONFIGURATION ---
# We use a slightly different T5 model that is better at generation
LLM_MODEL = 'MBZUAI/LaMini-Flan-T5-77M'
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
FAISS_INDEX_PATH = "km1139171_index.faiss"

# --- DATA PREPARATION (from your JSON) ---
# This function is now inside the app to be self-contained
def create_chunks_from_clean_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None
    
    fields = data.get("fields", {})
    document_parts = []

    for key, value in fields.items():
        key_cleaned = key.strip()
        if isinstance(value, list):
            value_str = ", ".join(str(v).strip() for v in value if v)
        else:
            value_str = str(value).strip()
        if value_str:
            document_parts.append(f"{key_cleaned}: {value_str}")
    
    full_document_text = "\n".join(document_parts)
    chunks = full_document_text.split('\n')
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# --- MODEL LOADING (Cached for performance) ---
@st.cache_resource
def load_models_and_index():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    try:
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    except:
        return None, None, None, None # Handle file not found gracefully
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    return embedding_model, faiss_index, tokenizer, model

# --- CORE RAG FUNCTION ---
def get_answer(question, documents, embedding_model, faiss_index, tokenizer, model):
    question_embedding = embedding_model.encode([question])
    
    # Search FAISS for the 3 most relevant chunks
    distances, indices = faiss_index.search(np.array(question_embedding).astype('float32'), k=3)
    
    # Combine the retrieved chunks into a single context
    retrieved_chunks = [documents[i] for i in indices[0]]
    context = "\n".join(retrieved_chunks)

    # This is the final, robust prompt
    prompt = f"""
    Please act as a friendly and helpful AIS customer service assistant.
    Answer the following question in a complete and polite Thai sentence based only on the provided information.

    Information:
    "{context}"

    Question:
    "{question}"

    Answer:
    """
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    output_sequences = model.generate(input_ids=inputs["input_ids"], max_length=200, num_beams=5, early_stopping=True)
    answer = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    return answer, context

# --- STREAMLIT UI ---
st.title("🤖 Chatbot ถาม-ตอบข้อมูลโปรโมชั่น")
st.write("ป้อนคำถามเกี่ยวกับโปรโมชั่น แล้ว Chatbot จะค้นหาข้อมูลและตอบคำถามให้คุณ")

# Load all necessary components
embedding_model, faiss_index, llm_tokenizer, llm_model = load_models_and_index()
documents = create_chunks_from_clean_json("KM1139171_clean.json")

if not all([embedding_model, faiss_index, llm_tokenizer, llm_model, documents]):
    st.error("เกิดข้อผิดพลาดในการโหลดโมเดลหรือไฟล์ข้อมูล กรุณาตรวจสอบว่าไฟล์ `km1139171_index.faiss` และ `KM1139171_clean.json` อยู่ใน repository")
else:
    user_question = st.text_input("คำถามของคุณ:")

    if st.button("ส่งคำถาม"):
        if user_question:
            with st.spinner("กำลังค้นหาข้อมูลและสร้างคำตอบ..."):
                answer, context = get_answer(user_question, documents, embedding_model, faiss_index, llm_tokenizer, llm_model)
                st.subheader("คำตอบ:")
                st.write(answer)
                with st.expander("ข้อมูลที่ใช้ในการตอบ (Context)"):
                    st.write(context)
        else:
            st.warning("กรุณาป้อนคำถาม")
