from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


data=TextLoader("document loaders/notes.txt") 
docs=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
)
chunk=splitter.split_documents(docs)


print(len(chunk))