# Conversation with window memory.
# Conversational Chatbot.

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.memory import ConversationBufferWindowMemory

llm = ChatOllama(model="gemma3:1b",temperature=0.2)

memory = ConversationBufferWindowMemory(
    memory_key="Chat_History",
    return_messages=True,
    k=3
)

prompt = PromptTemplate(
    input_variables=["Chat_History","user_input"],
    template="""
        You are a helpful assistant.
        Conversation so far:
        {Chat_History}

        User: {user_input}
        Assistant: 
"""
)

chain = prompt | llm

while True:
    user = input("User ---> ")

    if user.lower() == "exit":
        print("Bot ---> See you next time, bye!")
        break
    else:
        # load memory
        memory_var = memory.load_memory_variables({})

        response = chain.invoke({
            "Chat_History": memory_var["Chat_History"],
            "user_input": user
        })

        # save memory
        memory.save_context(
            {"user_input":user},
            {"Output":response.content}
        )

        print(f"Bot ---> {response.content}")




