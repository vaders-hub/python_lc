import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field

load_dotenv()


class CusineRecipe(BaseModel):
    name: str = Field(description="name of a cusine")
    recipe: str = Field(description="recipe to cook the cusine")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))


def llm_parser_comma():
    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template="List five {subject}.\n{format_instructions}",
        input_variables=["subject"],
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser

    result = chain.invoke({"subject": "popular Korean cusine"})

    print("\nChain Result:", result)


def llm_parser_json():
    output_parser = JsonOutputParser(pydantic_object=CusineRecipe)
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser

    result = chain.invoke({"query": "Let me know how to cook Bibimbap"})

    print("\nChain Result:", result)
