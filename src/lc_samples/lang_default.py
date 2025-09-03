import os
# import asyncio
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

# prompt + model + output parser
prompt = ChatPromptTemplate.from_template("지구과학에서 {topic}에 대해 간단히 설명해주세요.")
model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
output_parser = StrOutputParser()

# 2. 체인 생성
chain = prompt | model | output_parser


async def run_async():
    result = await chain.ainvoke({"topic": "해류"})
    print("ainvoke 결과:", result[:50], "...")


# asyncio.run(run_async())


def llm_test():
    # 3. invoke 메소드 사용
    # result = chain.invoke({"topic": "지구 자전"})
    # print("invoke 결과:", result)

    # batch 메소드 사용
    # topics = ["지구 공전", "화산 활동", "대륙 이동"]
    # results = chain.batch([{"topic": t} for t in topics])
    # for topic, result in zip(topics, results):
    #     print(f"{topic} 설명: {result[:50]}...")

    stream = chain.stream({"topic": "지진"})
    print("stream 결과:")
    for chunk in stream:
        print(chunk, end="", flush=True)
    print()
