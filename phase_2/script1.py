# LCEL --- LangChain Expression Language

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
        Explain {topic} in one short paragraph.
    """
)

llm = ChatOllama(
    model= "gemma3:1b",
    temperature=0.2
)

chain = prompt | llm

response = chain.invoke({"topic": "Python List"})
print(response.content)



