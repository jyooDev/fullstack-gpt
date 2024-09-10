from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
import os
import streamlit as st
from datetime import timedelta

    
@st.cache_data(show_spinner="Loading file...", ttl = timedelta(hours=1))
def load_split_docs(file, service):
    file_content = file.read()
    file_path = f"./.cache/files/{service}-gpt/{file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True) 
    with open(file_path, "wb") as f:
        f.write(file_content)
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = "\n",
        chunk_size = 600,
        chunk_overlap = 100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


def create_cache_dir(file = None, docs = None):
    if file:
        cache_dir_path = f"./.cache/embeddings/{file.name}"        
    os.makedirs(os.path.dirname(cache_dir_path), exist_ok=True) 
    return LocalFileStore(cache_dir_path) 


@st.cache_data(show_spinner="Embedding file...", ttl = timedelta(hours=1))
def embed_file(file, service, api_key):
    docs = load_split_docs(file, service)
    cache_dir = create_cache_dir(file)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever
