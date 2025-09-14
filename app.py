import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import asyncio
import httpx # For making API calls

# --- CONFIGURATION ---
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
JSON_FILE_PATH = "KM1139171_clean.json"
# Use the Gemini Flash model for speed, quality, and its large free tier.
GEMINI_API_MODEL = "gemini-1.5-flash-latest"

# --- DATA & RETRIEVAL SYSTEM SETUP (Cached for performance) ---

@st.cache_data
def load_and_chunk_document(file_path):
    """Loads and chunks the document from the clean JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        fields = data.get("fields", {})
        parts = []
        for key, value in fields.items():
            key_cleaned = key.strip()
            if isinstance(value, list):
                # Join list items into a single string
                value_str = ", ".join(str(v).strip() for v in value if v)
            else:
                value_str = str(value).strip()
            
            if value_str:
                parts.append(f"{key_cleaned}: {value_str}")
        
        # Split the combined text into chunks
        full_document_text = "\n".join(parts)
        chunks = [chunk.strip() for chunk in full_document_text.split('\n') if chunk.strip()]
        return chunks
    except Exception as e:
        st.error(f"Error loading or parsing {file_path}: {e}")
        return None

@st.cache_resource
def setup_retrieval_system(_documents):
    """Creates embeddings and the FAISS index from the document chunks."""
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = embedding_model.encode(_documents, show_progress_bar=False)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))
        return embedding_model, index
    except Exception as e:
        st.error(f"Error initializing the retrieval system: {e}")
        return None, None

# --- API COMMUNICATION ---
async def get_gemini_response(api_key, prompt):
    """Calls the Gemini API to get a conversational answer."""
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_API_MODEL}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=60.0)
            response.raise_for_status()
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
    except httpx.HTTPStatusError as e:
        return f"API Error: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- MAIN STREAMLIT APP ---
async def main():
    st.title("🤖 Chatbot ข้อมูลโปรโมชั่น (Powered by Gemini API)")
    st.write("แชทบอทนี้ใช้ข้อมูลจากเอกสารของคุณและตอบคำถามอย่างเป็นธรรมชาติด้วย Gemini API")

    # Load data and build the search index in memory
    documents = load_and_chunk_document(JSON_FILE_PATH)
    embedding_model, faiss_index = setup_retrieval_system(documents)

    if not documents or not embedding_model or not faiss_index:
        st.error("Application failed to initialize. Please check your source files and configuration on GitHub.")
        return

    # Get the user's API key from the sidebar
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("กรุณาใส่ Google API Key ของคุณ:", type="password", help="รับคีย์ฟรีได้ที่ Google AI Studio")
        if not api_key:
            st.warning("โปรดใส่ API Key ของคุณเพื่อเริ่มใช้งาน")
            st.stop()
        st.success("API Key ได้รับแล้ว, พร้อมใช้งาน!")

    # Chat interface
    user_question = st.text_input("คำถามของคุณเกี่ยวกับโปรโมชั่น:")

    if st.button("ส่งคำถาม"):
        if user_question:
            with st.spinner("กำลังค้นหาข้อมูลและสร้างคำตอบ..."):
                # 1. Retrieve relevant context using FAISS
                question_embedding = embedding_model.encode([user_question])
                _, indices = faiss_index.search(np.array(question_embedding).astype('float32'), k=5)
                context = "\n".join([documents[i] for i in indices[0]])

                # 2. Build the final prompt for the Gemini API
                prompt = f"""ในฐานะผู้ช่วยบริการลูกค้าของ AIS ที่เป็นมิตรและมีความรู้, จงตอบคำถามต่อไปนี้อย่างสุภาพและเป็นกันเองเป็นภาษาไทยที่สมบูรณ์: "{user_question}"
                
                จงใช้ข้อมูลต่อไปนี้ในการตอบเท่านั้น อย่าเพิ่มเติมข้อมูลที่ไม่มีในนี้:
                ---
                {context}
                ---
                """
                
                # 3. Get the answer from the Gemini API
                answer = await get_gemini_response(api_key, prompt)
                
                st.subheader("คำตอบ:")
                st.write(answer)
                with st.expander("ข้อมูลที่ใช้ในการตอบ (Context)"):
                    st.write(context)
        else:
            st.warning("กรุณาป้อนคำถาม")

if __name__ == "__main__":
    asyncio.run(main())

