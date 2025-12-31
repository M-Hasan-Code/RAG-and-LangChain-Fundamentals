from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Annotated
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore as lang_pinecone
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()

class QueryRequest(BaseModel):
    query: Annotated[str, Field(..., description="Enter your Query here...", max_length=500)]

pinecone_api = os.environ['PINECONE_API_KEY']
api_key = os.environ['OPENROUTER_API_KEY']

model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

try:
    embed_model = OllamaEmbeddings(model='all-minilm:22m')
    embed_model.embed_query("test query")
except Exception as e:
    raise Exception(f"Failed to initialize Ollama embeddings: {str(e)}")

index_name = "demo-vectorstore"
try:
    existing_vector_store = lang_pinecone.from_existing_index(index_name, embed_model)
except Exception as e:
    raise Exception(f"Failed to initialize Pinecone vector store: {str(e)}")

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent assistant for the University of Karachi. Answer the question based solely on the provided context. If the context lacks sufficient information, say "I don't have enough information to answer this question. Please visit https://www.uok.edu.pk/ for more details."

Context: {context}
Question: {question}
Answer in simple and concise language:
"""
)

base_retriever = existing_vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

def format_text(retrieved):
    return "\n".join([
        f"Page {doc.metadata.get('page_no', 'unknown')}: {doc.page_content}"
        for doc in retrieved
    ])

parallel_chain = RunnableParallel({
    "context": base_retriever | RunnableLambda(format_text),
    "question": RunnablePassthrough()
})

main_chain = parallel_chain | prompt_template | model | StrOutputParser()

@app.get('/')
def home():
    return {"Message": "RAG University of Karachi Chatbot"}

@app.post("/ask")
def ask_rag(request: QueryRequest):
    try:
        answer = main_chain.invoke({"question": request.query})
        return {"answer": answer}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")