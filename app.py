import os
import time
import tempfile
import unstructured
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredFileLoader,
    UnstructuredHTMLLoader, UnstructuredPowerPointLoader
)
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Streamlit session state
if "faiss_index" not in st.session_state:
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
        api_key=GOOGLE_API_KEY,
        model="models/embedding-001",
        task_type="retrieval_document"
    )
    st.session_state.faiss_index = None  # Initialize FAISS index to None

def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        extension = uploaded_file.name.split('.')[-1].lower()
        
        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        try:
            if extension == 'txt':
                loader = TextLoader(temp_file_path)
            elif extension in ['doc', 'docx']:
                loader = UnstructuredFileLoader(temp_file_path)
            elif extension == 'pdf':
                loader = PyPDFLoader(temp_file_path)
            elif extension in ['ppt', 'pptx']:
                loader = UnstructuredPowerPointLoader(temp_file_path)
            elif extension == 'html':
                loader = UnstructuredHTMLLoader(temp_file_path)
            else:
                st.error(f"Unsupported file format: {extension}")
                os.unlink(temp_file_path)  # Delete the temporary file
                continue
            
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
        finally:
            os.unlink(temp_file_path)  # Ensure the temporary file is deleted
    return documents

def split_docs_into_chunks(documents):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_documents(documents)

def create_faiss_vectorstore(documents, embeddings):
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def setup_retrieval_qa(faiss_index):
    llm = ChatGoogleGenerativeAI(
        api_key=GOOGLE_API_KEY,
        model="models/gemini-1.5-pro-latest",
        temperature=0.7,
        max_output_tokens=512
    )
    retriever = faiss_index.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Streamlit UI
st.title("🦜🔗 Langchain Document QA with FAISS")

# File Upload
uploaded_files = st.file_uploader(
    "Upload your documents (PDF, TXT, DOCX, HTML, PPTX)",
    accept_multiple_files=True
)

if uploaded_files:
    st.info("Processing uploaded files...")
    documents = load_documents(uploaded_files)
    if documents:
        chunked_docs = split_docs_into_chunks(documents)
        st.session_state.faiss_index = create_faiss_vectorstore(chunked_docs, st.session_state.embeddings)
        st.success("Documents added to vectorstore!")
    else:
        st.error("No valid text chunks to process.")

# Question Input
if st.session_state.faiss_index:
    qa_chain = setup_retrieval_qa(st.session_state.faiss_index)
    
    query = st.text_input("Ask a question:")
    if query:
        start = time.process_time()
        response = qa_chain({"query": query})
        end = time.process_time()

        # Display Response
        st.write("### Response:")
        st.write(response["result"])
        st.write(f"Time taken: {end - start:.2f} seconds")

        # Display Source Documents
        with st.expander("Source Documents:"):
            for i, doc in enumerate(response["source_documents"]):
                st.write(f"**Document {i + 1}**")
                st.write(doc.page_content)
                st.write("-" * 50)
else:
    st.warning("Upload documents to enable question answering.")



