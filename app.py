import streamlit as st
from transformers import pipeline

# --- MODEL LOADING (Cached for performance) ---
# We are using a model specifically for conversation
@st.cache_resource
def load_chatbot_model():
    try:
        # This pipeline is designed for text generation
        chatbot = pipeline("text2text-generation", model='MBZUAI/LaMini-Flan-T5-77M')
        return chatbot
    except Exception as e:
        st.error(f"Error loading the language model: {e}")
        return None

# --- STREAMLIT UI ---
st.title("🤖 Simple LLM Chatbot Test")
st.write("This is a simple chat to test if the AI model is working.")

# Load the model
chatbot = load_chatbot_model()

if chatbot:
    user_question = st.text_input("Your message:")

    if st.button("Send"):
        if user_question:
            with st.spinner("The AI is thinking..."):
                # We send the question directly to the model
                answer = chatbot(user_question, max_length=100, num_beams=5, early_stopping=True)
                
                st.subheader("AI's Response:")
                # The response is inside a list and a dictionary, so we extract it
                st.write(answer[0]['generated_text'])
        else:
            st.warning("Please type a message.")
else:
    st.error("The chatbot model could not be loaded. Please check the logs.")

