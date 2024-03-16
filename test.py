from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
# import ollama model
from langchain_community.llms import Ollama

client = Ollama(base_url="http://localhost:11434", model="mistral")

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OllamaEmbeddings(model='mistral')
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | client
    | StrOutputParser()
)

result = chain.invoke("where does harrison work")
print(result)

