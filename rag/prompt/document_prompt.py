from langchain.prompts import ChatPromptTemplate


def get_document_prompt():
    return ChatPromptTemplate.from_messages([    
        ("system", 
         "You are a helpful assistant. Answer questions using only the following context. "
         "If you don't know the answer, just say you don't know, don't make it up:\n\n{context}"),
        ("human", "{question}"),
    ])