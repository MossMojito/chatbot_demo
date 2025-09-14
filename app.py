import streamlit as st
from transformers import pipeline

# --- MODEL LOADING (Cached for performance) ---
# This model is small AND multilingual, designed for instructions.
LLM_MODEL = 'bigscience/mt0-small'

@st.cache_resource
def load_chatbot_model():
    try:
        # We use the text2text-generation pipeline
        chatbot = pipeline("text2text-generation", model=LLM_MODEL)
        return chatbot
    except Exception as e:
        st.error(f"Error loading the language model: {e}")
        return None

# --- STREAMLIT UI ---
st.title("🤖 Multilingual LLM Chatbot Test")
st.write("This chat tests a multilingual AI model. Please speak Thai to it.")

# Load the model
chatbot = load_chatbot_model()

if chatbot:
    user_question = st.text_input("Your message:")

    if st.button("Send"):
        if user_question:
            with st.spinner("AI is thinking..."):
                # A simple prompt is often effective for instruction-tuned models
                prompt = f"Q: {user_question} A:"
                
                # Generate the answer
                answer = chatbot(prompt, max_length=100)
                
                st.subheader("AI's Response:")
                st.write(answer[0]['generated_text'])
        else:
            st.warning("Please type a message.")
else:
    st.error("The chatbot model could not be loaded. Please check the logs.")

