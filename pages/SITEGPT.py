from constants.sitegpt_constant import cloudflare_sitemap_url, cloudflare_filter_urls
from rag.validation import validate_api_key
from rag.retriever.website_retriever import load_and_embed_website
from rag.prompt.site_prompt import answers_template, choose_prompt
from rag.model import get_openai_model, CallbackHandler
from rag.chat import send_message, paint_history
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import streamlit as st

# import asyncio
# asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


st.set_page_config(
    page_title="Site GPT",
    page_icon=":spider_web:",
    layout="wide",
)

if "valid_api_key" not in st.session_state:
    st.session_state["valid_api_key"] = False

st.title("Site GPT")

st.markdown(  """
Welcome! 
            
Ask questions about the following Cloudflare products: AI Gateway, Cloudflare Vectorize, and Workers AI.
    
Enter your OpenAI API Key to ask questions.
""")

def get_answers(inputs):
    docs = inputs["docs"]
    question= inputs["question"]
    llm = get_openai_model(api_key)
    answers_chain = answers_template | llm
    answers = []
    result = {"question": question,
              "answers": [{
        "answer": answers_chain.invoke({
            "question": question, "context": doc.page_content}),
        "source": doc.metadata["source"],
        "date": doc.metadata["lastmod"],
    } for doc in docs
    ]}
    return result

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    llm = get_openai_model(api_key, callback_handler=CallbackHandler.Chat)
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


with st.sidebar:
    docs = []
    api_key = st.text_input("OpenAI API Key", type="password")
    key_validation_button = st.button("Validate")
    if key_validation_button:
        st.session_state["valid_api_key"] = validate_api_key(api_key)
    
    st.link_button(
        label = "GitHub Code:computer:",
        url = "https://github.com/jyooDev/fullstack-gpt/blob/main/pages/SITEGPT.py"
    )

if st.session_state["valid_api_key"]:
    retriever = load_and_embed_website(api_key,cloudflare_sitemap_url, cloudflare_filter_urls)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask Anything! about your file...")
    if message:
        send_message(message, "human")
        chain = {"docs": retriever, "question": RunnablePassthrough()} | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
        try:                        
            with st.chat_message("ai"):
                chain.invoke(message)
        except Exception as e:
            send_message(f"Error occurred: {e}", "ai", save=True)
else:
    st.session_state["messages"] = []