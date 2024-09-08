from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import StreamingStdOutCallbackHandler
from rag.chat import save_message
import streamlit as st


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
def get_openai_model(api_key, callback_handler : str = "streamingStdOut"):
    if callback_handler == "streamingStdOut":
        callback_handler = StreamingStdOutCallbackHandler()
    elif callback_handler == "chat":
        callback_handler = ChatCallbackHandler()
    return ChatOpenAI(
        model = "gpt-4o-mini",
        temperature=0.1,
        streaming=True,        
        callbacks=[callback_handler],
        openai_api_key=api_key,
    )