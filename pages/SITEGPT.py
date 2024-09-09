from rag.validation import validate_api_key
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from rag.retriever import get_splitter
import streamlit as st
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


st.set_page_config(
    page_title="Site GPT",
    page_icon="📄",
    layout="wide",
)

if "valid_api_key" not in st.session_state:
    st.session_state["valid_api_key"] = False

st.title("Site GPT")

st.markdown(  """
Welcome!
            
Use this chatbot to ask questions about the content of the website!

Write the URL of the website on the sidebar.
""")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    key_validation_button = st.button("Validate")

    url = st.text_input("Enter URL", placeholder="https://example.com")

    if key_validation_button:
        st.session_state["valid_api_key"] = validate_api_key(api_key)
    if st.session_state["valid_api_key"]:
        a = 1

    
if url:
    loader = AsyncChromiumLoader(urls=[url])
    docs = loader.load()
    st.write(docs)
    html2text_transformer = Html2TextTransformer()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(transformed)