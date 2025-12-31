from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

template1= PromptTemplate(
    template= "Write a detailed summary (12-15)lines about the {topic}",
    input_variables=["topic"]
)
template2= PromptTemplate(
    template= "Write a 5 line Summary about {text}",
    input_variables=["text"]
)

parser =StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'Machine Learning theory'})

print(result)