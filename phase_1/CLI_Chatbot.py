# Command Line Input Chatbot(local LLM) -- mini project 1
from langchain_ollama import ChatOllama

# text = "hi!"

# llm = ChatOllama(
#     model="gemma3:1b",
#     temperature=0.3
# )
# output = llm.invoke(text)
# print(f"Bot: {output.content}")

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0.3
)

print("Type 'exit' to quit")
while True:
    user_input = input("You---> ")
    if user_input == "exit":
        break
    else:
        output = llm.invoke(user_input)
        print(f"Bot--> {output.content}")




