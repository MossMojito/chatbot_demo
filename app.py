import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import asyncio
import httpx

# --- CONFIGURATION ---
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
JSON_FILE_PATH = "KM1139171_clean.json"
GEMINI_API_MODEL = "gemini-1.5-flash-latest"

# --- DATA & RETRIEVAL SYSTEM SETUP (Cached for performance) ---

@st.cache_data
def load_and_chunk_document(file_path):
    """Loads and chunks the document from the clean JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        fields = data.get("fields", {})
        parts = [f"{k.strip()}: {', '.join(v) if isinstance(v, list) else v}" for k, v in fields.items()]
        text = "\n".join(parts)
        chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]
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
    st.title("🤖 Chatbot ข้อมูลโปรโมชั่น")
    st.write("แชทบอทที่จดจำการสนทนาได้ ขับเคลื่อนด้วย Gemini API")

    # Load data and build the search index in memory
    documents = load_and_chunk_document(JSON_FILE_PATH)
    embedding_model, faiss_index = setup_retrieval_system(documents)

    if not all([documents, embedding_model, faiss_index]):
        st.error("Application failed to initialize. Please check files on GitHub.")
        return

    # Get the user's API key from the sidebar
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("กรุณาใส่ Google API Key ของคุณ:", type="password", help="รับคีย์ฟรีได้ที่ Google AI Studio")
        if not api_key:
            st.warning("โปรดใส่ API Key ของคุณเพื่อเริ่มใช้งาน")
            st.stop()
        st.success("API Key ได้รับแล้ว, พร้อมใช้งาน!")

    # --- CHAT MEMORY SETUP ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- CHAT INPUT AND LOGIC ---
    if user_question := st.chat_input("สอบถามเกี่ยวกับโปรโมชั่นได้เลย..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาข้อมูลและสร้างคำตอบ..."):
                # 1. Retrieve relevant context using FAISS
                question_embedding = embedding_model.encode([user_question])
                _, indices = faiss_index.search(np.array(question_embedding).astype('float32'), k=3) # Use fewer chunks for more focus
                context = "\n".join([documents[i] for i in indices[0]])

                # 2. Get the last few messages for conversation history
                chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])

                # 3. Build a more advanced prompt with persona, instructions, context, and history
                prompt = f"""You are a friendly and knowledgeable AIS customer service assistant. Your goal is to answer the user's question in a helpful and conversational way, speaking in complete Thai sentences.

                Here is the recent conversation history:
                <history>
                {chat_history}
                </history>

                Here is the most relevant information from your knowledge base to answer the CURRENT user question:
                <context>
                {context}
                </context>

                Based on the conversation history and the provided context, answer the user's last message: "{user_question}"
                """
                
                # 4. Get the answer from the Gemini API
                answer = await get_gemini_response(api_key, prompt)
                st.markdown(answer)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    asyncio.run(main())

