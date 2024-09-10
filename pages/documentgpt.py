from rag.prompt.document_prompt import get_document_prompt
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import streamlit as st
from rag.retriever.file_retriever import embed_file
from rag.chat import send_message, format_docs, paint_history
from rag.model import get_openai_model, CallbackHandler
from rag.validation import validate_api_key

st.set_page_config(
    page_title="Document GPT",
    page_icon="📄",
    layout="wide",
)

if "valid_api_key" not in st.session_state:
    st.session_state["valid_api_key"] = False

prompt = get_document_prompt()

st.title("Document GPT")

st.markdown(  """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
""")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    key_validation_button = st.button("Validate")
    if key_validation_button:
        st.session_state["valid_api_key"] = validate_api_key(api_key)
        file = st.file_uploader("Upload a .txt, .pdf, .docs, .md files only", type=[
                            "pdf", "txt", "docx", "md"])
    st.link_button(
        label = "GitHub Link:computer:",
        url = "https://github.com/jyooDev/fullstack-gpt/blob/main/pages/documentgpt.py"
    )

if st.session_state["valid_api_key"] and file:
    retriever = embed_file(
        file=file, 
        service="document",
        api_key=api_key
    )
    llm = get_openai_model(api_key, callback_handler=CallbackHandler.Chat)
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