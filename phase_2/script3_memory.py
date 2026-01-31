# Memory (Chatbot with Memory)
# Conversation with Buffer Memory

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.memory import ConversationBufferMemory


llm = ChatOllama(
    model="gemma3:1b",
    temperature=0.2
)

memory = ConversationBufferMemory(
    memory_key="Chat_History",
    return_messages=True
)

prompt = PromptTemplate(
    input_variables=["Chat_History","UserInput"],
    template="""
        You are a helpful assistant.
        Conversation so far:
        {Chat_History}

        User: {UserInput}
        Assistant: 
"""
)

chain = prompt | llm

while True:
    user_input = input("User: ")

    if user_input.lower() == "exit":
        print("Bot: See you next time, bye!")
        break
    else:
        # load memory
        memory_variables = memory.load_memory_variables({})

        response = chain.invoke(
            {
                "Chat_History": memory_variables["Chat_History"],
                "UserInput": user_input
            }
        )

        # save memory
        memory.save_context(
            {"UserInput": user_input},
            {"Output": response.content}
        )
        print(f"Bot: {response.content}")
