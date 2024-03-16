from langchain.text_splitter import Language
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
# import ollama model
from langchain_community.llms import Ollama


repo_path = "/Users/andrewlaird/personal/LambKnuckleBones"

loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*.py",  # Adjusted pattern to include both directories
    suffixes=[".py"],
    exclude=["**/site-packages", "**/venv", "**/env", "**/node_modules"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)

documents = loader.load()
print(len(documents))


python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(texts, embedding=OllamaEmbeddings(model='mistral'))
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

client = Ollama(base_url="http://localhost:11434", model="mistral")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | client
    | StrOutputParser()
)

result = chain.invoke("How do you know when a knuckle bones game is over")
print("-------")
print(result)

