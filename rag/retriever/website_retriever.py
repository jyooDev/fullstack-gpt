import streamlit as st
from datetime import timedelta
from bs4 import BeautifulSoup
from typing import Optional, List
from langchain.document_loaders import SitemapLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def parse_cloudflare_page(content:BeautifulSoup) -> str:
    header = content.find("header")
    footer = content.find("footer")
    for element in [header, footer]:
        if element is not None:
            element.decompose()    

    return str(content.find("main").get_text().replace("\n", " ").replace("Cloudflare DashboardDiscordCommunityLearning CenterSupport Portal  Cookie Settings", ""))

@st.cache_data(show_spinner="Loading website...")
def load_and_embed_website(api_key, url:str, filter_urls: Optional[List[str]] = None):
    loader = SitemapLoader(
        web_path=url, 
        filter_urls=filter_urls,
        parsing_function=parse_cloudflare_page)
    loader.requests_per_second = 2
    loader.requests_kwargs = {"verify": False}
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 1000,
        chunk_overlap = 200,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    return retriever