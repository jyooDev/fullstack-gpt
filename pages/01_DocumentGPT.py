import time
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import streamlit as st


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

@st.cache_data(show_spinner="Embedding file...") #if the file is same, then the streamlit will skip running the function 
def embed_file(file):
    file_content = file.read()
    new_file_path = f"./.cache/files/{file.name}"
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    with open(new_file_path, "wb") as f:
        f.write(file_content)

    cache_dir_path = f"./.cache/embeddings/{file.name}"    
    os.makedirs(os.path.dirname(cache_dir_path), exist_ok=True)
    cache_dir = LocalFileStore(cache_dir_path)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 100,
    )

    loader = UnstructuredFileLoader(new_file_path)

    #does not work from here...
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(documents=docs, embedding=cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!
"""
)

file = st.file_uploader(
    "Upload a .txt .pdf or .docx file",
    type=["pdf", "txt", "docx"],
)
# with st.sidebar:
#     file = st.file_uploader(
#         "Upload a .txt .pdf or .docx file",
#         type=["pdf", "txt", "docx"],
#     )

if file:
    retriever = embed_file(file)
    messages = st.chat_input()
    s = retriever.invoke("winston")
    st.write(s)