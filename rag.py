from uuid import uuid4


from dotenv import load_dotenv
from pathlib import Path
from langchain_classic.chains import RetrievalQAWithSourcesChain
#from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import  RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import os
# Load environment variables from the .env file (like API keys)
load_dotenv()

# Define constants for text chunking and vector storage
CHUNK_SIZE = 1000
COLLECTION_NAME = 'real_estate'
VECTORSTORE_DIR = Path(__file__).parent/"resourses/vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Global variables to hold our Language Model and Vector Database instances
llm = None
vector_store = None

def initialize_components():
    """Initialize the Large Language Model and the Vector Database."""
    global llm, vector_store

    # Initialize the LLM (Groq Llama 3) if it hasn't been set up yet
    if llm is None:
        llm = ChatGroq(model='llama-3.3-70b-versatile', temperature= 0.9, max_tokens= 500)

    # Initialize the Vector Store (ChromaDB) if it hasn't been set up yet
    if vector_store is None:
        # Define the embedding function used to convert text into mathematical vectors
        ef = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL,
                                   model_kwargs = {"trust_remote_code": True})

        # Set up the Chroma vector store with persistent local storage
        vector_store = Chroma(
            collection_name= COLLECTION_NAME,
             embedding_function= ef,
            persist_directory=str(VECTORSTORE_DIR),
        )





def process_urls(urls):
    """Fetch content from URLs, chunk the text, and store it in the vector database."""
    try:
        yield "Initializing components"
        initialize_components()

        yield "Resetting vector store"
        # Clear out any previous data in the collection to ensure fresh results for new URLs
        vector_store.reset_collection()

        yield "Scraping and loading data from URLs"
        # Load the documents from the provided web URLs using WebBaseLoader for maximum reliability
        loader = WebBaseLoader(
            web_path=urls,
            header_template={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
            continue_on_failure=True
        )
        data = loader.load()

        if not data:
            yield "Error: Could not retrieve any content from the provided URLs."
            return

        yield "Splitting text into chunks"
        # Set up the text splitter to divide large documents into smaller, semantic chunks for the LLM
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ' '],
            chunk_size=CHUNK_SIZE
        )
        docs = text_splitter.split_documents(data)

        if not docs:
            yield "Error: No text chunks could be extracted from the documents."
            return

        # Generate unique IDs for each individual text chunk
        uuids = [str(uuid4()) for _ in range(len(docs))]
        
        yield "Adding chunks to vector database"
        # Add the document chunks to the Chroma vector store so they can be searched later
        vector_store.add_documents(docs, ids=uuids)
        
        yield "Success: URLs processed successfully!"

    except Exception as e:
        yield f"Error processing URLs: {str(e)}"


def generate_answer(query):
    """Retrieve relevant context from the vector store and generate an answer using the LLM."""
    if not vector_store:
        raise RuntimeError("VectorDB is not initilaized")

    # Set up the retrieval-augmented generation chain to link the LLM and the Vector DB
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever= vector_store.as_retriever())
    
    # Invoke the chain with the user's question
    result = chain.invoke({'question':  query},
                          return_only_outputs=True,)
                          
    # Extract the sources from the response provided by the LLM
    sources = result.get("sources", "")

    return result['answer'], sources




if __name__ == "__main__":
    initialize_components()
    url = [
        "https://www.morganstanley.com/insights/articles/mortgage-rates-forecast-2025-2026-will-mortgage-rates-go-down",
        "https://www.freedommortgage.com/learn/market-updates/housing-market-outlook"]

    process_urls(url)

    results = vector_store.similarity_search(
        "30 years morgage rate",
        k=2,

    )

    print(results)
    answer, sources = generate_answer("what is the national average for 30 year fixed mortgage rate as of mid-april 2026"
                        )
    print(f" Answer : {answer}\n\n Sources : {sources}")

    print("*********************")
    context = ''.join([doc.page_content for doc in results])
    print(context)