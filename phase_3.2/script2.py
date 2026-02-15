# Query Rewriting, Rag+llm

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

document = TextLoader("phase_3.1/documnet_1.txt").load()

chunks = RecursiveCharacterTextSplitter(separators=["\n\n"],chunk_size=1000,chunk_overlap=20).split_documents(document)

embed = OllamaEmbeddings(model="nomic-embed-text")

vecStore = Chroma.from_documents(
    documents=chunks,
    embedding=embed,
    collection_name="collection2_3.2",
    persist_directory="./phase_3.2/VS2_3.2"
)

retriver = vecStore.as_retriever(search_kwargs={"k":2})

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

llm = ChatOllama(model="gemma3:1b",temperature=0)

memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

rewrite_prompt = ChatPromptTemplate.from_template(
    """
    Given the conversation history and a follow-up question,
    rewrite the question to be a standalone question.

    Conversation:
    {chat_history}

    Follow-up Question:
    {question}

    Standalone Question:
"""
)

prompt = ChatPromptTemplate.from_template(
    """
        You are a helpful AI assistant.

        Conversation History:
        {chat_history}

        Context from documents:
        {context}

        User Question:
        {question}

        If the answer is not in the context, say "I don't know."
"""
)

rewrite_chain = (

    {
        "chat_history": lambda _: memory.load_memory_variables({})["chat_history"],
        "question": RunnablePassthrough()
    } | rewrite_prompt | llm | StrOutputParser()

)

main_chain = (
    {
        "chat_history": lambda _: memory.load_memory_variables({})["chat_history"],
        "context": rewrite_chain | retriver | format_docs,
        "question": RunnablePassthrough()
    } | prompt | llm | StrOutputParser()
)

while True:
    user_question = input("User ---> ")
    if user_question.lower() == "exit":
        print("See you again, bye!")
        break
    else:
        response = main_chain.invoke(user_question)
        memory.save_context({"input":user_question},{"output":response})
        print(f"Bot ---> {response}")


