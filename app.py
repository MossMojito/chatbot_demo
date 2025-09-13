import streamlit as st
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Configuration ---
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
LLM_MODEL = 'google/mt5-small'

# !!! EDIT THESE TWO VARIABLES !!!
FAISS_INDEX_PATH = "km1139171_index.faiss"  # <--- 1. Put your FAISS file name here
documents = [
    # <--- 2. Paste your list of documents here, exactly as used to create the index
    'Topic: โปรโมชั่นภายในประเทศ (อัตรา/เงื่อนไข)',
    'Topic: โปรโมชั่นภายในประเทศ (อัตรา/เงื่อนไข)',
    'Sub Topic: แพ็กเกจเสริมรายเดือน (Postpaid) สมัครรายครั้ง (One-Time)',
    'Sub Topic: แพ็กเกจเสริมเติมเงิน (Prepaid) สมัครรายครั้ง (One-Time)',
    'More Info1: แพ็กเสริมเน็ต',
    'More Info1: แพ็กเสริมเน็ต',
    # ... (paste all your other document strings here) ...
]
# !!! END OF EDITING SECTION !!!


@st.cache_resource
def load_models():
    """Loads the embedding and language models."""
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    return embedding_model, llm_tokenizer, llm_model

embedding_model, llm_tokenizer, llm_model = load_models()

try:
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
except Exception as e:
    st.error(f"Could not load the FAISS index file '{FAISS_INDEX_PATH}'. Make sure the file is in the same folder as app.py.")
    st.stop()

def answer_question(question, top_k=3):
    """Finds relevant docs and generates an answer using an LLM."""
    question_embedding = embedding_model.encode([question])
    distances, indices = faiss_index.search(np.array(question_embedding).astype('float32'), top_k)

    retrieved_docs = [documents[i] for i in indices[0]]
    context = " ".join(retrieved_docs)

    prompt = f"Based on this information: --- {context} --- Answer the question: {question}"

    input_ids = llm_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).input_ids
    outputs = llm_model.generate(input_ids, max_length=256, num_beams=5, early_stopping=True)
    answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, context

# --- Streamlit User Interface ---
st.title("🤖 Chatbot ถาม-ตอบข้อมูลโปรโมชั่น")
st.write("ป้อนคำถามเกี่ยวกับโปรโมชั่น แล้ว Chatbot จะค้นหาข้อมูลและตอบคำถามให้คุณ")

user_question = st.text_input("คำถามของคุณ:")

if st.button("ส่งคำถาม"):
    if user_question:
        with st.spinner("กำลังค้นหาข้อมูลและสร้างคำตอบ..."):
            answer, context = answer_question(user_question)
            st.subheader("คำตอบ:")
            st.write(answer)
            with st.expander("ข้อมูลที่ใช้ในการตอบ (Context)"):
                st.write(context)
    else:
        st.warning("กรุณาป้อนคำถาม")