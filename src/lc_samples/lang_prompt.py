import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

load_dotenv()


llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY")
)


def llm_propmt():
    template_text = "안녕하세요, 제 이름은 {name}이고, 나이는 {age}살입니다."
    prompt_template = PromptTemplate.from_template(template_text)
    filled_prompt = prompt_template.format(name="개발자", age=30)
    combined_prompt = (
        prompt_template
        + PromptTemplate.from_template("\n\n보도방 다 문닫게 해주세요.")
        + "\n\n{language}로 번역해주세요."
    )
    chain = combined_prompt | llm | StrOutputParser()
    result = chain.invoke({"age": 30, "language": "영어", "name": "개발자"})

    print("result", result)


def llm_chat_prompt():
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "이 시스템은 천문학 질문에 답변할 수 있습니다."),
            ("user", "{user_input}"),
        ]
    )

    messages = chat_prompt.format_messages(
        user_input="태양계에서 가장 큰 행성은 무엇인가요?"
    )

    print("messages", messages)


def llm_chat_message_prompt():
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "이 시스템은 천문학 질문에 답변할 수 있습니다."
            ),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ]
    )

    messages = chat_prompt.format_messages(
        user_input="태양계에서 가장 큰 행성은 무엇인가요?"
    )
    chain = chat_prompt | llm | StrOutputParser()

    result = chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})

    print("result", result)


def llm_chat_few_prompt():
    examples = [
        {
            "question": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?",
            "answer": "지구 대기의 약 78%를 차지하는 질소입니다.",
        },
        {
            "question": "광합성에 필요한 주요 요소들은 무엇인가요?",
            "answer": "광합성에 필요한 주요 요소는 빛, 이산화탄소, 물입니다.",
        },
        {
            "question": "피타고라스 정리를 설명해주세요.",
            "answer": "피타고라스 정리는 직각삼각형에서 빗변의 제곱이 다른 두 변의 제곱의 합과 같다는 것입니다.",
        },
        {
            "question": "지구의 자전 주기는 얼마인가요?",
            "answer": "지구의 자전 주기는 약 24시간(정확히는 23시간 56분 4초)입니다.",
        },
        {
            "question": "DNA의 기본 구조를 간단히 설명해주세요.",
            "answer": "DNA는 두 개의 폴리뉴클레오티드 사슬이 이중 나선 구조를 이루고 있습니다.",
        },
        {
            "question": "원주율(π)의 정의는 무엇인가요?",
            "answer": "원주율(π)은 원의 지름에 대한 원의 둘레의 비율입니다.",
        },
    ]
    example_prompt = PromptTemplate.from_template("질문: {question}\n{answer}")
    prompt = FewShotPromptTemplate(
        examples=examples,  # 사용할 예제들
        example_prompt=example_prompt,  # 예제 포맷팅에 사용할 템플릿
        suffix="질문: {input}",  # 예제 뒤에 추가될 접미사
        input_variables=["input"],  # 입력 변수 지정
    )
    result = prompt.invoke(
        {"input": "화성의 표면이 붉은 이유는 무엇인가요?"}
    ).to_string()

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,  # 사용할 예제들
        OpenAIEmbeddings(),  # 임베딩 모델
        Chroma,  # 벡터 저장소
        k=1,  # 선택할 예제 수
    )

    # 새로운 질문에 대해 가장 유사한 예제를 선택합니다.
    question = "화성의 표면이 붉은 이유는 무엇인가요?"
    selected_examples = example_selector.select_examples({"question": question})
    print(f"입력과 가장 유사한 예제: {question}")
    for example in selected_examples:
        print("\n")
        for k, v in example.items():
            print(f"{k}: {v}")


def llm_chat_few_prompt_fix():
    # 예제 정의
    examples = [
        {
            "input": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?",
            "output": "질소입니다.",
        },
        {
            "input": "광합성에 필요한 주요 요소들은 무엇인가요?",
            "output": "빛, 이산화탄소, 물입니다.",
        },
    ]

    # 예제 프롬프트 템플릿 정의
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    # Few-shot 프롬프트 템플릿 생성
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    # 최종 프롬프트 템플릿 생성
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 과학과 수학에 대해 잘 아는 교육자입니다."),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    # 모델과 체인 생성
    chain = final_prompt | llm

    # 모델에 질문하기
    result = chain.invoke({"input": "지구의 자전 주기는 얼마인가요?"})

    print("llm_chat_few_prompt_fix", result.content)


def llm_chat_few_prompt_fix_dyn():
    # 더 많은 예제 추가
    examples = [
        {
            "input": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?",
            "output": "질소입니다.",
        },
        {
            "input": "광합성에 필요한 주요 요소들은 무엇인가요?",
            "output": "빛, 이산화탄소, 물입니다.",
        },
        {
            "input": "피타고라스 정리를 설명해주세요.",
            "output": "직각삼각형에서 빗변의 제곱은 다른 두 변의 제곱의 합과 같습니다.",
        },
        {
            "input": "DNA의 기본 구조를 간단히 설명해주세요.",
            "output": "DNA는 이중 나선 구조를 가진 핵산입니다.",
        },
        {
            "input": "원주율(π)의 정의는 무엇인가요?",
            "output": "원의 둘레와 지름의 비율입니다.",
        },
    ]

    # 벡터 저장소 생성
    to_vectorize = [" ".join(example.values()) for example in examples]
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

    # 예제 선택기 생성
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )

    # Few-shot 프롬프트 템플릿 생성
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_selector=example_selector,
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        ),
    )

    # 최종 프롬프트 템플릿 생성
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 과학과 수학에 대해 잘 아는 교육자입니다."),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    # 모델과 체인 생성
    chain = final_prompt | llm

    # 모델에 질문하기
    result = chain.invoke("태양계에서 가장 큰 행성은 무엇인가요?")
    print(result.content)
