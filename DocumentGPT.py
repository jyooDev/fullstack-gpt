import os
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
import streamlit as st


if "api_key" not in st.session_state:
    st.session_state["api_key"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = [] 

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
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



@st.cache_data(show_spinner="Embedding file...")  
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
        chunk_size=1000,
        chunk_overlap=100,
    )


    loader = UnstructuredFileLoader(new_file_path)
    docs = loader.load_and_split(text_splitter=splitter)


    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["api_key"])
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    
    vectorstore = FAISS.from_documents(documents=docs, embedding=cached_embeddings)
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
        send_message(
            message["message"],
            message["role"],
            save=False,
        )



def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# Sidebar
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )


    api_key = st.text_input("Enter API_KEY").strip()
    button = st.button("SAVE")
    if button:
        if api_key == "":
            st.warning("Enter API Key.")
        else:
            st.session_state["api_key"] = api_key 
            st.write("API KEY is saved.")


# Main content
if file and st.session_state["api_key"]:
    
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        streaming=True,
        callbacks={
            ChatCallbackHandler(),
        },
        openai_api_key=st.session_state["api_key"],
    )

    
    prompt = ChatPromptTemplate.from_messages([    
        ("system", 
         "You are a helpful assistant. Answer questions using only the following context. "
         "If you don't know the answer, just say you don't know, don't make it up:\n\n{context}"),
        ("human", "{question}"),
    ])

    
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()

   
    message = st.chat_input("Ask questions about your file...")
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
