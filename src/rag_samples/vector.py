import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
embeddings_model = OpenAIEmbeddings(api_key=api_key)

embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)


def test_chroma_vector_txt():
    loader = TextLoader('./src/static/history.txt')
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250,
        chunk_overlap=50,
        encoding_name='cl100k_base'
    )

    texts = text_splitter.split_text(data[0].page_content)

    db = Chroma.from_texts(
        texts, 
        embeddings_model,
        collection_name='history',
        persist_directory='./db/chromadb',
        collection_metadata={'hnsw:space': 'cosine'}, # l2 is the default
    )

    query = '누가 한글을 창제했나요?'
    docs = db.similarity_search(query)

    print(docs[0].page_content)


def test_chroma_vector_pdf():
    loader = PyMuPDFLoader('./src/static/kakao.pdf')
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
        encoding_name='cl100k_base'
    )

    docs = text_splitter.split_documents(data)

    db2 = Chroma.from_documents(
        docs, 
        embeddings_model,
        collection_name='esg',
        persist_directory='./db/chromadb',
        collection_metadata={'hnsw:space': 'cosine'},
    )

    query = '카카오뱅크의 환경목표와 세부추진내용을 알려줘?'
    docs = db2.similarity_search(query)
    mmr_docs = db2.max_marginal_relevance_search(query, k=4, fetch_k=10)

    # print(docs[-1].page_content)
    print(mmr_docs[-1].page_content)


def test_faiss_vector_pdf():
    loader = PyMuPDFLoader('./src/static/kakao.pdf')
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
        encoding_name='cl100k_base'
    )

    docs = text_splitter.split_documents(data)

    vectorstore = FAISS.from_documents(
        docs,
        embedding=embeddings_model,
        distance_strategy=DistanceStrategy.COSINE
    )

    query = '카카오뱅크가 중대성 평가를 통해 도출한 6가지 중대 주제는 무엇인가?'
    docs = vectorstore.similarity_search(query)
    mmr_docs = vectorstore.max_marginal_relevance_search(query, k=4, fetch_k=10)

    vectorstore.save_local('./db/faiss')

    # print(len(mmr_docs))
    print(mmr_docs[-1].page_content)


def test_faiss_vector_pdf_rtvr():
    loader = PyMuPDFLoader('./src/static/kakao.pdf')
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
        encoding_name='cl100k_base'
    )

    docs = text_splitter.split_documents(data)

    vectorstore = FAISS.from_documents(
        docs,
        embedding=embeddings_model,
        distance_strategy=DistanceStrategy.COSINE
    )

    query = '카카오뱅크의 환경목표와 세부추진내용을 알려줘'

    # 가장 유사도가 높은 문장을 하나만 추출
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 5, 'fetch_k': 50}
    )

    docs = retriever.get_relevant_documents(query)

    print(docs[0])
