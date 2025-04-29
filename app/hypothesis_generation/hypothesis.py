
from app.prompts.hypothesis_prompt import hypothesis_format_prompt
import os

hypothesis_endpoint = os.getenv('HYPOTHESIS_ENDPOINT')
class Hypothesis_generation:

    def __init__(self, llm) -> None:
        self.llm = llm
        pass

    def get_by_enrich_id(self, id):
        # summary = hypothesis_endpoint.get(enrich_id=id)
        # return summary
        pass

    def format_user_query(self, query):
        # prompt = hypothesis_format_prompt.format(query=query)
        # response = self.llm.generate(prompt)
        # return response
        pass

    def generate_hypothesis(self):
        # user_query = self.format_user_query()
        # param = {phenotype = user_query["phenotype"] , variant = user_query["variant"]}
        # response = hypothesis_endpoint.get()
        # return response
        pass

    