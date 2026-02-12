# Question -> Retrieve text -> answer
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama

loader = TextLoader("phase_3.1/documnet_1.txt")
document = loader.load()

chunks = RecursiveCharacterTextSplitter(separators=["\n\n"],chunk_size=1000,chunk_overlap=0).split_documents(document)

print("---------Chunks---------")
for i,c in enumerate(chunks):
    print(f"Chunk {i}")
    print(c.page_content)
    print(c.metadata)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorStore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="KnowledgeBase2",
    persist_directory="./phase_3.1/ChromaDB_Vectorstore2"
)

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0.1
)

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

retriver = vectorStore.as_retriever(search_kwargs={"k":1})
q = "Which specific C++ standard introduced 'smart pointers' to help with safer memory management?"
results = retriver.invoke(input=q)
print("------------Retrived chunks------------")
for doc in results:
    print(doc.page_content)


chain = prompt|llm

response = chain.invoke(
    {
        "context": [d.page_content for d in results],
        "question": q
    }
)

print("------------Final answer------------")
print(response.content)




