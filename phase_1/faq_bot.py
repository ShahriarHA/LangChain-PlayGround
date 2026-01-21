# Prompt-Driven FAQ Bot -- mini project 2

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

faq_pmt = PromptTemplate(
    input_variables=["question"],
    template="""
        You are an FAQ assistant.
        Answer the question clearly.
        If you do not know, say: "I don't know".


        Question: {question}
    """
)

question = "What is LangChain?"

q = faq_pmt.format(question=question)
print(q)

faq_llm = ChatOllama(
    model="gemma3:1b",
    temperature=0.0
)

response = faq_llm.invoke(q)
print(response.content)

