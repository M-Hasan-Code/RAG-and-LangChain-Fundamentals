from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Step 1: Load the ML.txt file
loader = TextLoader("ML.txt",  encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    # chunk_overlap = 20
)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)