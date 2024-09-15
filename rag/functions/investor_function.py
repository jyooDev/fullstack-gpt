from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
import json

def dgg_search_docs(inputs):
    query = inputs["query"]
    ddg = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())
    return ddg.run(query)

def wiki_search_docs(inputs):
    query = inputs["query"]
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wikipedia.invoke(query)


functions_map = {
    "dgg_search_docs": dgg_search_docs,
    "wiki_search_docs": wiki_search_docs,
}


functions = [
    {
        "type": "function",
        "function": {
            "name": "dgg_search_docs",
            "description": "Given the query returns relevant documents using DuckDuckGo search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A query to search on DuckDuckGo",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_search_docs",
            "description": "Given the query returns relevant documents using wikipedia search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A query to search on DuckDuckGo",
                    }
                },
                "required": ["query"],
            },
        },
    },
]