import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
embeddings_model = OpenAIEmbeddings(api_key=api_key)


def load_embeddings():
    embeddings = embeddings_model.embed_documents(
        [
            '안녕하세요!',
            '어! 오랜만이에요',
            '이름이 어떻게 되세요?',
            '날씨가 추워요',
            'Hello LLM!'
        ]
    )
    print('embedding', len(embeddings))