import os
from dotenv import load_dotenv
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))


def get_current_season():
    month = datetime.now().month
    if 3 <= month <= 5:
        return "봄"
    elif 6 <= month <= 8:
        return "여름"
    elif 9 <= month <= 11:
        return "가을"
    else:
        return "겨울"


def llm_partial():

    prompt = PromptTemplate(
        template="{season}에 일어나는 대표적인 지구과학 현상은 {phenomenon}입니다.",
        input_variables=["phenomenon"],  # 사용자 입력이 필요한 변수
        partial_variables={"season": get_current_season},  # 함수를 통해 동적으로 값을 생성하는 부분 변수
    )

    print(prompt.format(phenomenon="꽃가루 증가"))
