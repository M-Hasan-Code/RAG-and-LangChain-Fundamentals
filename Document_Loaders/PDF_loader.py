from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Lab 14 to 17.pdf")

pdf = loader.load()  # lazy_load

print(len(pdf))

for pages in pdf:
    print(pages.page_content)