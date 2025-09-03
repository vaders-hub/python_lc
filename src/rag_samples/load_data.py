import os
from dotenv import load_dotenv
import bs4

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model='gpt-4o-mini', temperature=0, api_key=api_key)

url = "https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8"
loader = WebBaseLoader(url)
docs = loader.load()

    
def load_web_data():
    loader = WebBaseLoader(url)

    docs = loader.load()

    print(len(docs))
    print(len(docs[0].page_content))
    print(docs[0].page_content[5000:6000])


def load_and_split():
    loader = WebBaseLoader(url)

    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(api_key=api_key)
    )

    docs = vectorstore.similarity_search("격하 과정에 대해서 설명해주세요.")
    print(len(docs))
    print(docs[0].page_content)


def load_and_gen():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(api_key=api_key)
    )

    template = '''Answer the question based only on the following context:
    {context}
       
    Question: {question}
    '''

    prompt = ChatPromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever()
    
    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)

    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # Chain 실행
    response = rag_chain.invoke("격하 과정에 대해서 설명해주세요.")
    
    print(response)  # 생성된 답변 출력
    
    
def load_web():
    url1 = "https://blog.langchain.dev/customers-replit/"
    url2 = "https://blog.langchain.dev/langgraph-v0-2/"

    loader = WebBaseLoader(
        web_paths=(url1, url2),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("article-header", "article-content")
            )
        )
    )
    docs = loader.load()
    len(docs)
    
    print(docs[0].page_content[:500])
    