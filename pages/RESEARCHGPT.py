from rag.validation import validate_api_key
from rag.chat import send_message, paint_history
from rag.functions.investor_function import *
from rag.assistants.investor_assistant import *
from openai import OpenAI
import streamlit as st
import time

st.set_page_config(
    page_title="Research GPT",
    page_icon=":spider_web:",
    layout="wide",
)

if "valid_api_key" not in st.session_state:
    st.session_state["valid_api_key"] = False

st.title("Investor GPT")

st.markdown(  """
Welcome! 
            
Ask questions about publicly traded companies. I can assist you  to decide if you should buy the stock or not:)
""")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    key_validation_button = st.button("Validate")
    if key_validation_button:
        st.session_state["valid_api_key"] = validate_api_key(api_key)
    
    st.link_button(
        label = "GitHub Code:computer:",
        url = "https://github.com/jyooDev/fullstack-gpt/blob/main/pages/SITEGPT.py"
    )

if st.session_state["valid_api_key"]:
    client = OpenAI(api_key=api_key)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
        
    message = st.chat_input("Ask for an advise for a stock of a company of your interest!")
    if message:
        send_message(message, "human")
        if "assistant" not in st.session_state:
            print("Creating Assistant...")
            assistant = client.beta.assistants.create(
                name = "Research assistant",
                temperature=0.1,
                description="You help users do research with two search engines: DuckDuckGo and Wikipedia.",
                model="gpt-4o-mini",
                tools=functions
            )
            st.session_state["assistant"] = assistant
            print("Creating Thread...")
            thread = client.beta.threads.create()
            st.session_state["thread"] = thread
        else:
            assistant = st.session_state["assistant"]
            thread = st.session_state["thread"]
        print("Sending message to Thread...")    
        send_message_to_thread(thread.id, message, client)
        print("Creating run...")
        run = client.beta.threads.runs.create(
            thread_id = thread.id,
            assistant_id = assistant.id
        )

        while get_run(run.id, thread.id, client).status in ["queued", "in_progress"]:
            print(get_run(run.id, thread.id, client).status)
            time.sleep(1)
        while get_run(run.id, thread.id, client).status in ["required_action"]:
            submit_tool_outputs(run.id, thread.id, client)
            time.sleep(1)
        message = get_messages(thread.id, client)
        send_message(message, "ai")

else:
    st.session_state["messages"] = []