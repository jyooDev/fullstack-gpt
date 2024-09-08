import os
import openai
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import streamlit as st
from rag.retriever import embed_file
from rag.chat import save_message, send_message
from rag.model import get_openai_model

st.set_page_config(
    page_title="Document GPT",
    page_icon="📄",
    layout="wide",
)
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
            retriever = embed_file(
                file = file, 
                service = "document",
                api_key= api_key
            )
            llm = get_openai_model()
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