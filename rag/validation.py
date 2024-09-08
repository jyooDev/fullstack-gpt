import openai
import os
import streamlit as st

@st.cache_resource
def validate_api_key(api_key):
    openai.api_key = api_key
    try:
        openai.Model.list()
    except openai.error.AuthenticationError as e:
        st.warning("Invalid API Key")
        api_key = None
        return False
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("Valid API Key")
        return True