from langchain.retrievers import WikipediaRetriever
from datetime import timedelta
import streamlit as st
@st.cache_data(show_spinner="Searching Wikipedia...", ttl = timedelta(hours=1))
def get_docs_from_wikipedia_retriever(topic):
    retriever = WikipediaRetriever(top_k_results=5)
    return retriever.get_relevant_documents(topic)