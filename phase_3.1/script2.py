# Question -> retrive text -> answer
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


loader = TextLoader("phase_3.1/knowledgeBase.txt")
documents = loader.load()

spliter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
chunks = spliter.split_documents(documents)

# for i, chunk in enumerate(chunks):
#     print(f"chunk {i}")
#     print(chunk.page_content)
#     print(chunk.metadata)


# call ollama embeddings: This converts text into vectors using a local embedding model.
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# store chunks in vectorstores
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./phase_3.1/vectorstore",
    collection_name="my_collection"
)

# retrive relevant chunks
retriver = vectorstore.as_retriever(kwargs={"k":2})

query = "What is RAG?"

results = retriver.invoke(query)

for doc in results:
    print(doc.page_content)



