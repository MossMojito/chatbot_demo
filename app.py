import streamlit as st
import faiss
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# --- CONFIGURATION ---
QA_MODEL = 'deepset/roberta-base-squad2'
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
FAISS_INDEX_PATH = "km1139171_index.faiss"

# --- DATA PREPARATION ---
@st.cache_data
def load_document_chunks(file_path="KM1139171_clean.json"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        st.error(f"Error: Could not read {file_path}. Make sure it's in your GitHub repository.")
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

# --- MODEL AND INDEX LOADING ---
@st.cache_resource
def load_all_models_and_index():
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        qa_pipeline = pipeline("question-answering", model=QA_MODEL, tokenizer=QA_MODEL)
        return embedding_model, faiss_index, qa_pipeline
    except Exception as e:
        st.error(f"Error loading models or index file: {e}. Please check your files and requirements.txt.")
        return None, None, None

# --- CORE LOGIC ---
def find_answer(question, documents, embedding_model, faiss_index, qa_pipeline):
    question_embedding = embedding_model.encode([question])
    distances, indices = faiss_index.search(np.array(question_embedding).astype('float32'), k=5) # Search 5 chunks for better context
    
    retrieved_chunks = [documents[i] for i in indices[0]]
    context = "\n".join(retrieved_chunks)

    result = qa_pipeline(question=question, context=context)
    return result['answer'], context

# --- STREAMLIT UI ---
st.title("🤖 Chatbot ค้นหาข้อมูลโปรโมชั่น")
st.write("ป้อนคำถามเกี่ยวกับโปรโมชั่น แล้ว Chatbot จะค้นหาคำตอบที่ตรงที่สุดจากเอกสารมาให้คุณ")

documents = load_document_chunks()
embedding_model, faiss_index, qa_pipeline = load_all_models_and_index()

if documents and qa_pipeline:
    user_question = st.text_input("คำถามของคุณ:")

    if st.button("ส่งคำถาม"):
        if user_question:
            with st.spinner("กำลังค้นหาคำตอบ..."):
                answer, context = find_answer(user_question, documents, embedding_model, faiss_index, qa_pipeline)
                st.subheader("คำตอบ:")
                st.success(f"**{answer}**") # Display the answer clearly
                with st.expander("ข้อมูลที่ใช้ในการตอบ (Context)"):
                    st.write(context)
        else:
            st.warning("กรุณาป้อนคำถาม")
