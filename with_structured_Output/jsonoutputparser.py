from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
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

parser = JsonOutputParser()

template = PromptTemplate(
    template="What is the Name,age,product and warranty in {text}\n{format_instruction}",
    input_variables=["text"],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
response = chain.invoke({"text":"My name is Muhammad Hasan and i am 19 years old i buy this product with 4 year warranty"}) 
print(response)