from langchain_community.document_loaders import WebBaseLoader


url = [
    "https://www.amazon.in/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/9355421982",
    "https://www.amazon.in/Python-Data-Analysis-Wrangling-Grayscale/dp/9355421907/ref=pd_lpo_d_sccl_1/258-0847684-4070028?pd_rd_w=lZTC2&content-id=amzn1.sym.e0c8139c-1aa1-443c-af8a-145a0481f27c&pf_rd_p=e0c8139c-1aa1-443c-af8a-145a0481f27c&pf_rd_r=BD9SYKBC5ZF4K76K550K&pd_rd_wg=2xNbL&pd_rd_r=edd7f953-f00f-4e18-bf5b-c4db90ccbc06&pd_rd_i=9355421907&psc=1"
]
loader = WebBaseLoader(url)
docs = loader.load()

print(len(docs))
print(docs[0].page_content)