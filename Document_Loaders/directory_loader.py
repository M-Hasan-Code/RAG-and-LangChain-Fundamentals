from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader

loader = DirectoryLoader(
    path="Directory",
    glob="**/*.pdf",           # Only load .pdf files
    loader_cls=PyPDFLoader,    # Use the correct PDF loader
    show_progress=True
)

docs = loader.lazy_load()

for doc in docs:
    print(doc.page_content)
