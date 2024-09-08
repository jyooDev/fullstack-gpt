import streamlit as st
from rag.output_parsers import JsonOutputParser
from rag.chat import format_docs
import json

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty, _chain):
    chain =  _chain
    response = chain.invoke(
        {
        "context": format_docs(_docs),
        "difficulty": difficulty
    }
    )
    response = response.additional_kwargs["function_call"]["arguments"]
    response = json.loads(response)["questions"]
    return response
