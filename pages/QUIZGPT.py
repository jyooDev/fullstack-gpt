import streamlit as st
from rag.retriever.file_retriever import load_split_docs
from rag.retriever.wikipedia_retriever import get_docs_from_wikipedia_retriever
from rag.functions.quiz_function import function
from rag.model import get_openai_model, CallbackHandler
from rag.prompt.quiz_prompt import get_question_prompt
from rag.validation import validate_api_key
from rag.chat import format_docs
import json

st.set_page_config(
    page_title = "QuizGPT",
    page_icon = "",
    layout = "centered",
)

if "valid_api_key" not in st.session_state:
    st.session_state["valid_api_key"] = False


st.title("QuizGPT")


with st.sidebar:
    docs = None
    topic = None
    api_key = st.text_input("Enter your API key.", type="password")
    key_validation_button = st.button("Validate")
    if key_validation_button:
        st.session_state["valid_api_key"] = validate_api_key(api_key)
    if st.session_state["valid_api_key"]:
        choice = st.selectbox("Choose what you want to use.", (
            "File",
            "Wikipedia Article",
        ))
        if choice == "File":
            file = st.file_uploader("Upload a .txt, .pdf, .docs, .md files only", type=[
                                "pdf", "txt", "docx", "md"])
            if file:
                docs = load_split_docs(
                file = file,
                service = "quiz",
                )
        else:
            topic = st.text_input("Search Wikipedia..").lower()
            if topic:
                docs = get_docs_from_wikipedia_retriever(topic)
        difficulty = st.radio(
            "Select the Quiz Difficulty",
            ["Easy", "Intermediate", "Hard"])
    st.link_button(
        label = "GitHub Link:computer:",
        url = "https://github.com/jyooDev/fullstack-gpt"
    ) 
    

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


if not docs or not api_key:
    st.markdown(
        """
    Welcome to QuizGPT!
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge.
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )    
else:     
    llm = get_openai_model(api_key, CallbackHandler.StreamingStdOut).bind(
        function_call={
            "name": "generate_quiz",
        },
        functions=[
            function,
        ]
    )
    question_prompt = get_question_prompt()
    chain = question_prompt | llm
    response = run_quiz_chain(
        _docs=docs,
        topic=topic if topic else file.name,
        difficulty = difficulty,
        _chain = chain,)
    with st.form("questions_form"):
        answer_ct = 0
        for question in response:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                answer_ct += 1
            elif value is not None:
                st.error("Wrong!")

        button = st.form_submit_button()   
        if button and answer_ct == 10:
            st.balloons()
            st.header("Congratulation!")
            answer_ct = 0
        elif button and answer_ct != 10:
            st.header(f"Score: {answer_ct}/10")
            answer_ct = 0