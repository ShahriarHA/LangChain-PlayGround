# Task 1: PromptTemplate Practice

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
        Just write a short description about {topic} in 3 bullet points. You must generate only one line of information in every 3 bullet points.

"""
)

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0.2
)

chain = prompt | llm

response = chain.invoke({"topic":"C++ programming language"})
# print(response.content)

print("---split lines---")
lines = response.content.splitlines()
# print(lines)
print(f"len of lines: {len(lines)}")

updated_lines = []
for line in lines:
    # print(line)
    if "*" in line:
        updated_lines.append(line)
# print(updated_lines)  
print(f"---final output is---")
[print(i) for i in updated_lines]


