from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from pydantic import BaseModel, Field
from typing import Optional


class LeaseTypes(BaseModel):
    lease: bool = Field(False, description="Include standard leases")
    ownership: bool = Field(False, description="Include ownership leases")
    sale_leaseback: bool = Field(False, description="Include sale leaseback leases")
    sublease: bool = Field(False, description="Include subleases")
    equipment_lease: bool = Field(False, description="Include equipment leases")

class AdhocOptions(BaseModel):
    display_approved_analyses: Optional[bool] = Field(description="Display approved analyses")
    # required field
    lease_types: LeaseTypes

class PydanticLLMModel():
    def __init__(self, pydantic_object: BaseModel) -> None:
        self.pydantic_object = pydantic_object
        self.client = Ollama(base_url="http://localhost:11434", model="codellama")
        self.parser = PydanticOutputParser(pydantic_object=self.pydantic_object)
        self.prompt = PromptTemplate(
            template="Generate JSON matching the query according to the format instructions. You are configuring a report, true means that the data should be included in the report and false means that it should not. If you are unsure include extra information. Use UTF8 encoding. Include only the JSON response without commentary or code blocks. Format instructions and query details are provided below.\nFormat Instructions: {format_instructions}\nQuery: {query}",
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self.max_retries = 3

    def parse_with_retry(self, client_result):
        retries = self.max_retries
        while retries > 0:
            try:
                return self.parser.parse(client_result)
            except Exception as e:
                prompt_with_error = client_result + "\n Please fix these errors and return the correct JSON: \n" + str(e)
                print(f"Error parsing response: {e}")
                retries -= 1
                if retries > 0:
                    print(f"Retrying...")
                    client_result = self.client.invoke(prompt_with_error)
                    print("-----")
                    print(f"New client result: {client_result}")
                else:
                    raise OutputParserException(f"Failed to parse response after retries")

    def invoke(self, query):
        prompt_result = self.prompt.invoke({"query": query})
        print(prompt_result)
        print("-----")
        client_result = self.client.invoke(prompt_result)
        print(client_result)
        print("-----")
        try:
            return self.parse_with_retry(client_result)
        except OutputParserException as e:
            return self.pydantic_object()

        
# query = "Can you give equipment proposals and ownership leases, only in progress leases"
# model = PydanticLLMModel(pydantic_object=AdhocOptions)
# result = model.invoke(query)
# print(result)
