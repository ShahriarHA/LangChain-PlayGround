# Question -> retrive text -> answer
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

loader = TextLoader("phase_3.1/documnet_1.txt")
document = loader.load()

spliter = RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=20)
chunks = spliter.split_documents(document)

# for i, chukn in enumerate(chunks):
#     print(f"Chunk {i}")
#     print(chukn.page_content)
#     print(chukn.metadata)

embedding = OllamaEmbeddings(model="nomic-embed-text")

vectorStore = Chroma.from_documents(
    documents=chunks,
    collection_name="KnowledgeBase1",
    embedding=embedding,
    persist_directory="./phase_3.1/ChromaDB_Vetorstore1"

)

retriver = vectorStore.as_retriever(kwargs={"k":2})

q = "What was the original name of C++ when it was first developed in 1979?"
q2 = "Which specific C++ standard introduced 'smart pointers' to help with safer memory management?"
results = retriver.invoke(input=q2)
for doc in results:
    print(doc.page_content)


# RAG prompt + LLM
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="gemma3:1b",temperature=0.1)

prompt = PromptTemplate(
    input_variables=["context","question"],
    template="""
        Answer the question using ONLY the context below.
        If the answer is not in the context, say "I don't know".
        context:
        {context}
        question:
        {question}
"""
)

context = "\n\n".join([doc.page_content for doc in results])

chain = prompt|llm

response = chain.invoke({
    "context":context,
    "question":q2
})

print("-------Final response-------")
print(response.content)











