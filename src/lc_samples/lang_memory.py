import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

store = {}
config = {"configurable": {"session_id": "astronomy_chat_1"}}
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def llm_memory():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 천문학 전문가입니다. 사용자와 친근한 대화를 나누며 천문학 질문에 답변해주세요."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,  # 기본 체인 (prompt | llm)
        get_session_history,  # 세션 히스토리를 가져오는 함수
        input_messages_key="question",  # 사용자 입력 키
        history_messages_key="history",  # 대화 기록 키
    )

    response1 = chain_with_history.invoke(
        {"question": "안녕하세요, 저는 지구과학을 공부하는 학생입니다."}, config=config  # type: ignore
    )
    print("\nChain response1:", response1)

    response2 = chain_with_history.invoke({"question": "태양계에서 가장 큰 행성은 무엇인가요?"}, config=config)  # type: ignore
    print("\nChain response1:", response2)

    response3 = chain_with_history.invoke({"question": "그 행성의 위성은 몇 개나 되나요?"}, config=config)  # type: ignore
    print("\nChain response1:", response3)
