# Question → Retrieve text → Answer

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("phase_3.1/knowledgeBase.txt")
documents = loader.load()

spliter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

chunks = spliter.split_documents(documents)
# print(len(chunks))
# print(chunks)
for i, chunk in enumerate(chunks):
    print(f"Chunk : {i}")
    print(print(f"{chunk.page_content}"))

# load and split documents