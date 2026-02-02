# Task 2: LCEL Practice

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# User input → Prompt → LLM → Print output
llm = ChatOllama(
    model="gemma3:1b",
    temperature=0.2
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
        Write a short description of the following question.
        {question}
"""
)

# chain = prompt | llm

# response = chain.invoke({"question":"What is Computer Science?"})
# print(response.content)

# User input → Rewrite → Answer → Print

user_question = PromptTemplate(
    input_variables=["q"],
    template="""
        Rewrite the following question only once.
        {q}
"""
)
rewritten_question_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
        Consider the following content. In this content you'll find a question; generate an answer for this.
        {content}
"""
)

chain = user_question | llm
original_question = "When I should sit for IELTS exam?"
rewritten_question = chain.invoke({"q": original_question})
print("--- written question content ---")
print(rewritten_question.content)

chain2 = rewritten_question_prompt | llm
solution = chain2.invoke({"content":rewritten_question.content})

print("--- solution of the content ---")
print(solution.content)

