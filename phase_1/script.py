# First langchain script
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0.2
)

# response = llm.invoke("explain AI in one sentence.")
response1 = llm.invoke("can you tell me how deeply you can help me in IT filed?")
print(response1.content)



