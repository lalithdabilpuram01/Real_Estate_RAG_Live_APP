from uuid import uuid4


from dotenv import load_dotenv
from pathlib import Path
from langchain_classic.chains import RetrievalQAWithSourcesChain
#from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import  RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import os
load_dotenv()

CHUNK_SIZE = 1000
COLLECTION_NAME = 'real_estate'
VECTORSTORE_DIR = Path(__file__).parent/"resourses/vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

llm = None
vector_store = None

def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model='llama-3.3-70b-versatile', temperature= 0.9, max_tokens= 500)

    if vector_store is None:

        ef = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL,
                                   model_kwargs = {"trust_remote_code": True})

        vector_store = Chroma(
            collection_name= COLLECTION_NAME,
             embedding_function= ef,
            persist_directory=str(VECTORSTORE_DIR),

        )





def process_urls(urls):
    yield "Initializing component"

    initialize_components()

    yield "resetting vector store"

    vector_store.reset_collection()


    loader = UnstructuredURLLoader(urls= urls)

    yield "loading data"
    data = loader.load()

    yield "splitting text into chunks"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ' '],
        chunk_size= CHUNK_SIZE
        )
    docs = text_splitter.split_documents(data)

    uuids = [str(uuid4()) for _ in range(len(docs))]
    yield "adding docs to vector"
    vector_store.add_documents(docs, ids= uuids)


def generate_answer(query):
    if not vector_store:
        raise RuntimeError("VectorDB is not initilaized")

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever= vector_store.as_retriever())
    result = chain.invoke({'question':  query},
                          return_only_outputs=True,)
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