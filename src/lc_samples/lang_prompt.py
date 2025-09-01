import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate,  HumanMessagePromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

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
    result = chain.invoke({"age":30, "language":"영어", "name":"개발자"})

    print('result', result)

def llm_chat_prompt():
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "이 시스템은 천문학 질문에 답변할 수 있습니다."),
        ("user", "{user_input}"),
    ])

    messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")

    print('messages', messages)

def llm_chat_message_prompt():
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("이 시스템은 천문학 질문에 답변할 수 있습니다."),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ]
    )

    messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
    chain = chat_prompt | llm | StrOutputParser()

    result = chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})

    print('result', result)
