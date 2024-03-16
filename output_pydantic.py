from langchain.output_parsers import PydanticOutputParser, RetryOutputParser
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

# Create an ollama model and set its output to the MyModel
client = Ollama(base_url="http://localhost:11434", model="codellama")

parser = PydanticOutputParser(pydantic_object=AdhocOptions)
retry_parser = RetryOutputParser.from_llm(parser=parser, llm=client, max_retries=3)
prompt = PromptTemplate(
    template="Generate JSON matching the query according to the format instructions. You are configuring a report, true means that the data should be included in the report and false means that it should not. If you are unsure include extra information. Use UTF8 encoding. Include only the JSON response without commentary or code blocks. Format instructions and query details are provided below.\nFormat Instructions: {format_instructions}\nQuery: {query}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

query = "Can you give equipment proposals and ownership leases, only in progress leases"


prompt_result = prompt.invoke({"query": query})
print(prompt_result)
print("-----")
client_result = client.invoke(prompt_result)
print(client_result)
print("-----")
def parse_with_retry(parser, client, client_result, max_retries):
    if max_retries > 0:
        try:
            return parser.parse(client_result)
        except Exception as e:
            prompt_with_error = client_result + "\n Please fix these errors and return the correct JSON: \n" + str(e)
            print(f"Error parsing response: {e}")
            print(f"Retrying...")
            client_result = client.invoke(prompt_with_error)
            print("-----")
            print(f"New client result: {client_result}")
            return parse_with_retry(parser, client, client_result, max_retries - 1)
    raise OutputParserException(f"Failed to parse response after {max_retries} retries")

result = parse_with_retry(parser, client, client_result, 3)
print(result);
