import streamlit as st 
import os 
import asyncio
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()
# Ensure event loop exists
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
# load the groq api and gemini api key
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY')

st.title("RAG Document Q&A With Groq and Gemini")
llm= ChatGroq(groq_api_key = groq_api_key, model= 'gemma2-9b-it')
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions bsed on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# def create_vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
#         st.session_state.loader = PyPDFDirectoryLoader("researchPapers")
#         st.session_state.docs = st.session_state.loader.load()
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

def create_vector_embedding():
    # Path to persist FAISS index
    index_path = "faiss_index"

    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

        # If FAISS index already exists, load it (no new API calls)
        if os.path.exists(index_path):
            st.session_state.vectors = FAISS.load_local(index_path, st.session_state.embeddings, allow_dangerous_deserialization=True)
            st.write("Loaded existing FAISS index (no new API calls).")

        else:
            # Load only 2 documents for now
            st.session_state.loader = PyPDFDirectoryLoader("researchPapers")
            st.session_state.docs = st.session_state.loader.load()

            # Split into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:2]
            )

            # Create FAISS index (this calls embeddings API ONCE)
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )

            # Save FAISS index for reuse
            st.session_state.vectors.save_local(index_path)
            st.write("Created and saved new FAISS index.")

user_prompt = st.text_input("Enter your query from the research paper:")

if st.button("Documents Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

import time 
if user_prompt:
    document_chain =create_stuff_documents_chain(llm, prompt)
    retriver= st.session_state.vectors.as_retriever()
    retrival_chain = create_retrieval_chain(retriver, document_chain)

    start = time.process_time()
    response = retrival_chain.invoke({"input": user_prompt})
    print(f"Response time: {time.process_time()-start}")

    st.write(response['answer'])
    # with a streamlit expander
    with st.expander("Documents similarity search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----------------------------")



# vec = embeddings.embed_query("What's the meaning of life?")
# print("Embedding (first 5 dims):", vec[:5])




