from langchain.prompts import ChatPromptTemplate


answers_template = ChatPromptTemplate.from_template(
"""
Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                
Then, give a score to the answer between 0 and 5.

If the answer answers the user question the score should be high, else it should be low.

Make sure to always include the answer's score even if it's 0.

Context: {context}
                                                
Examples:
                                                
Question: How far away is the moon?
Answer: The moon is 384,400 km away.
Score: 5
                                                
Question: How far away is the sun?
Answer: I don't know
Score: 0
                                                
Your turn!

Question: {question}
"""
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use only one answer that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            
            Make sure to line break between answer and source.
            Do not include the date when answering the user's question.
            Do not put **Answer:** in front of the answer.
            
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)