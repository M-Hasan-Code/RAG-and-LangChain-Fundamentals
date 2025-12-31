from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
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

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic and cover in 10 words'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic and cover in 10 words'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic and cover in 10 words'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

response = chain.invoke({"topic": "football"})

print(response)