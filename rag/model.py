from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.pydantic_v1 import Field
from rag.chat import save_message
import streamlit as st
from enum import Enum

class CallbackHandler(Enum):
    Chat = 1
    StreamingStdOut = 2

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_data
def get_openai_model(api_key, callback_handler : CallbackHandler | None = None):
    
    if callback_handler is None:
        return ChatOpenAI(
        model = "gpt-4o-mini",
        temperature=0.1,
        openai_api_key=api_key,
    )

    match callback_handler:
        case CallbackHandler.StreamingStdOut:
            handler = StreamingStdOutCallbackHandler()
        case CallbackHandler.Chat:
            handler = ChatCallbackHandler()

    return ChatOpenAI(
        model = "gpt-4o-mini",
        temperature=0.1,
        streaming=True,        
        callbacks=[handler],
        openai_api_key=api_key,
    )