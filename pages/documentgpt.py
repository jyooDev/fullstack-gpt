import os
import openai
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="Document GPT",
    page_icon="📄",
    layout="wide",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)



@st.cache_resource
def get_openai_model(api_key):
    return ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        openai_api_key=api_key,
    )


@st.cache_data(show_spinner="Embedding File...")
def embed_file(file, api_key):
    file_path = f"./.cache/files/{file.name}"  
    cache_dir_path = f"./.cache/embeddings/{file.name}"    
    os.makedirs(os.path.dirname(file_path), exist_ok=True) 
    os.makedirs(os.path.dirname(cache_dir_path), exist_ok=True)  
    cache_dir = LocalFileStore(cache_dir_path) 

    file_content = file.read()
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages([    
        ("system", 
         "You are a helpful assistant. Answer questions using only the following context. "
         "If you don't know the answer, just say you don't know, don't make it up:\n\n{context}"),
        ("human", "{question}"),
    ])

def validate_api_key(api_key):
    openai.api_key = api_key
    try:
        openai.Model.list()
    except openai.error.AuthenticationError as e:
        return False
    else:
        return True


st.title("Document GPT")

st.markdown(  """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
""")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    file = st.file_uploader("Upload a .txt, .pdf, .docs, .md files only", type=[
                            "pdf", "txt", "docx", "md"])

if api_key:
    if validate_api_key(api_key):
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("Valid API Key")
        if file:
            retriever = embed_file(file, api_key)
            llm = ChatOpenAI(
                temperature=0.1,
                streaming=True,
                callbacks=[ChatCallbackHandler()],
                openai_api_key=api_key,
            )
            send_message("I'm ready! Ask away!", "ai", save=False)
            paint_history()
            message = st.chat_input("Ask Anything! about your file...")
            if message:
                send_message(message, "human")
                
                chain = (
                    {
                        "context": retriever | RunnableLambda(format_docs),
                        "question": RunnablePassthrough(),
                    }
                    | prompt
                    | llm
                )
                try:                        
                    with st.chat_message("ai"):
                        chain.invoke(message)
                except Exception as e:
                    send_message(f"Error occurred: {e}", "ai", save=False)
   
else:
    st.session_state["messages"] = []