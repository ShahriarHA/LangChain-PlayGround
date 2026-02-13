# RAG + Memory
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferMemory

document = TextLoader("phase_3.1/documnet_1.txt").load()

chunks = RecursiveCharacterTextSplitter(separators=["\n\n"],chunk_size=1000,chunk_overlap=0).split_documents(document)

embedd = OllamaEmbeddings(model="nomic-embed-text")
VectorStore = Chroma.from_documents(
    documents=chunks,
    embedding=embedd, persist_directory="./phase_3.2/VS_3.2",collection_name="collection1_3.2"
)

retriver = VectorStore.as_retriever(search_kwargs={"k":2})

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

llm = ChatOllama(model="gemma3:1b",temperature=0)

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

memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

rag_chain = (
    {
        "context": retriver | format_docs,
        "question":RunnablePassthrough(),
        "chat_history": lambda _: memory.load_memory_variables({})["chat_history"]
    } | prompt | llm | StrOutputParser()
)

# chatBOT
while True:
    user = input("User ---> ")
    if user.lower() == "exit":
        print("See you again, bye!")
        break
    else:
        response = rag_chain.invoke(user)
        memory.save_context(
            {"input":user},{"output":response}
        )

        print(f"Bot ---> {response}")





