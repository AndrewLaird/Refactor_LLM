from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language

print(RecursiveCharacterTextSplitter.get_separators_for_language(Language.PHP))
