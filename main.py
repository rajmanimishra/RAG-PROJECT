
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# 2. Load existing vector database
vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

# 3. Create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5,
    }
)

# 4. Initialize Grok LLM
llm = ChatGroq(model="openai/gpt-oss-20b",)

# 5. Create chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant for a RAG application.

Answer the user's question using only the provided context.

Instructions:
1. Give accurate and relevant answers based on the context.
2. If the answer is unavailable, say:
   "I don't know the answer based on the provided documents."
3. Do not invent facts or use outside knowledge.
4. Explain in simple, clear language.
5. Use bullet points when helpful.

Context:
{context}"""
    ),
    ("human", "{question}")
])

# 6. Connect prompt with LLM
chain = chat_prompt | llm

print("RAG System Created!")
print("Press 0 to exit.")

while True:
    query = input("\nYou: ").strip()

    if query == "0":
        break

    if not query:
        continue

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    if not docs:
        print("\nAI: I don't know the answer based on the provided documents.")
        continue

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Generate answer
    response = chain.invoke({
        "context": context,
        "question": query
    })

    print("\nAI:", response.content)
