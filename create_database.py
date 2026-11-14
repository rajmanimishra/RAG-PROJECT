
"""
Steps:
1. Load the .txt file
2. Split the text into chunks
3. Generate embeddings
4. Store the chunks and embeddings in ChromaDB
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. Load the text file
loader = TextLoader(
    "document loaders/notes.txt",
    encoding="utf-8"
)
docs = loader.load()

print(f"Loaded documents: {len(docs)}")

# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print(f"Total chunks created: {len(chunks)}")

# 3. Initialize the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# 4. Store chunks and embeddings in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

print("Database created successfully!")
print("Documents embedded and stored in ChromaDB.")
