import streamlit as st

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

        
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)