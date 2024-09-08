from langchain.schema import BaseOutputParser
import json


class JsonOutputParser(BaseOutputParser):
    def parse(self, text: str) -> json:
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)