﻿# LangChain Document QA System

This application allows users to upload documents (PDF, TXT, DOCX, HTML, CSV, XML, PPTX), which are then processed into a vector store. You can interact with the system by asking questions, and the system will retrieve the most relevant documents to answer the queries using language models.

## Features

- Upload documents in various formats (PDF, TXT, DOCX, PPTX, HTML, CSV, XML).
- Automatic conversion of documents into chunks and embedding of content for efficient retrieval.
- Integration with a Google-based language model (Gemini) to answer questions.
- Displaying relevant source documents for each response.

## Requirements

- Python 3.8+
- `streamlit`
- `langchain`
- `langchain_community`
- `langchain_google_genai`
- `faiss-cpu`
- `openai`
- `nltk`
- `python-dotenv`

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
* Create a .env file in the root of the project and add your API keys:
```bash
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

## Usage
1. Run the Streamlit app:

```bash
streamlit run app.py
```
2. Once the app is running, navigate to http://localhost:8501 in your browser.

3. Upload your documents (PDF, TXT, DOCX, HTML, PPTX, etc.) using the file uploader.

4. Ask a question, and the system will retrieve the most relevant information from the uploaded documents.

5. View the source documents from which the answer was derived using the "Source Documents" expander.

## How it Works
1. Document Upload: Users upload documents, which are processed into text and converted into chunks.
2. Embedding: The text chunks are embedded using the specified language model (Gemini API or OpenAI).
3. Indexing: The embeddings are indexed using FAISS for fast retrieval.
4. Querying: When the user submits a query, the system retrieves the most relevant document chunks from the FAISS vector store and answers the question based on the context.
5. Source Documents: The application shows the source documents used to generate the answer.
## File Formats Supported
- PDF
- TXT
- DOCX
- PPTX
- HTML
- XML
- CSV

## Example
* Upload a document (e.g., a research paper in PDF format).
* Enter a query related to the document, such as "What are the key findings in this paper?"
* The system will retrieve the relevant sections and provide an answer, along with the document from which the answer was derived.

## Notes
* This app uses the Google Gemini API for question answering.
* FAISS is used as the vector database to store document embeddings and perform efficient similarity search.
* Ensure that you have the appropriate API keys and .env file configured.

## Live Demo
This application is deployed on Render. You can access it here : [Langchain Retrieval QA](https://langchain-retrieval-qa.onrender.com).
