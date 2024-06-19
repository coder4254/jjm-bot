import streamlit as st
import os
from dotenv import load_dotenv
from io import StringIO
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from googletrans import Translator

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Constants
emb_model = "models/embedding-001"
llm_model = "gemini-1.0-pro"
CHUNK_SIZE = 512
OVERLAP_SIZE = 128
CHAIN_TYPE = "stuff"

# Translator
translator = Translator()

def translate_text(text, target_language):
    if target_language == 'en':
        return text
    translation = translator.translate(text, dest=target_language)
    return translation.text

def get_vector_store(text_chunks):
    print("LOG: GET VECTOR STORE")
    embeddings = GoogleGenerativeAIEmbeddings(model=emb_model)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_local")

def get_conversational_chain():
    print("LOG: GET CONVERSATIONAL CHAIN")
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in provided context just say, "Answer is not available in the context"\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model=llm_model, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type=CHAIN_TYPE, prompt=prompt)
    return chain

def get_conversational_chain_2():
    print("LOG: GET CONVERSATIONAL CHAIN - 2")
    prompt_template = """
    Firstly go through the whole text. Answer the question in details, make sure to provide all the details,
    Accordingly format the answer in paragraph and points\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model=llm_model, temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type=CHAIN_TYPE, prompt=prompt)
    return chain

# Initialize chat history and logs
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me Anything from Operational Guidelines"
        }
    ]

if "language" not in st.session_state:
    st.session_state.language = "en"

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def user_input(user_question):
    print("LOG: USER_INPUT()")
    embeddings = GoogleGenerativeAIEmbeddings(model=emb_model)
    new_db = FAISS.load_local("faiss_local", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question})

    msg = "Answer is not available in the context"
    if response["output_text"] == msg:
        st.write("Finding the most relevant response....")
        chain = get_conversational_chain_2()
        response = chain({"input_documents": docs, "question": user_question})
        output_text = translate_text(response["output_text"], st.session_state.language)
        with st.chat_message("assistant"):
            st.markdown(output_text)
    else:
        output_text = translate_text(response["output_text"], st.session_state.language)
        with st.chat_message("assistant"):
            st.markdown(output_text)

    # Storing the User Message
    st.session_state.messages.append({"role": "user", "content": user_question})
    # Storing the Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": output_text})

def main():
    # Sidebar for language selection
    st.sidebar.header("Settings")
    language = st.sidebar.selectbox(
        "Select Language",
        ("English", "Hindi", "Kannada" , "Tamil", "Telugu", "Bengali", "Marathi", "Gujarati", "Malayalam", "Punjabi")
    )

    language_code = {
        "English": "en",
        "Hindi": "hi",
        "Kannada": "kn",
        "Tamil": "ta",
        "Telugu": "te",
        "Bengali": "bn",
        "Marathi": "mr",
        "Gujarati": "gu",
        "Malayalam": "ml",
        "Punjabi": "pa"
    }

    st.session_state.language = language_code[language]

    # Take user inputs
    user_question = st.chat_input("What's up?")
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)    
        user_input(user_question)

    with st.sidebar:
        st.header("JAL JEEVAN MISSION INDIA")
        st.title("A Chatbot for queries related to Jal Jeevan Mission.")

        # Create button for instructions
        if st.sidebar.button("Instructions"):
            st.markdown("Instructions for using the app:")
            st.markdown("1. The model is designed to find the exact answer from the document. If a perfect answer is not available, then an AI-based answer will be delivered to the user.")
            st.markdown("2. If the model doesn't follow the chain, try asking the full query.")
            st.markdown("3. If the model seems to be hallucinating, try refreshing the page or clearing the cache (of the Streamlit app).")

        # Footer in sidebar
        st.sidebar.markdown("<br>" * 1, unsafe_allow_html=True)
        st.sidebar.markdown("-" * 20)
        st.sidebar.markdown("Developed by: [Chandan Kumar](mailto:chandankr014@gmail.com)")
        st.sidebar.markdown("© JJM - IIM Bangalore Cell")

# Calling main
if __name__ == "__main__":
    main()
