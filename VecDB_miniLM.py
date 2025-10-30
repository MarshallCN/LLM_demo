from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from pathlib import Path


# === 文档加载 ===
def load_documents(folder: str):
    docs = []
    for path in Path(folder).rglob("*"):
        if path.suffix == ".txt":
            docs += TextLoader(str(path), encoding="utf-8").load()
        elif path.suffix == ".pdf":
            docs += PyPDFLoader(str(path)).load()
    return docs

# === 文本分块 ===
def split_docs(docs, chunk_size=512, chunk_overlap=64):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# === 向量化模型 ===
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === 主流程 ===
raw_docs = load_documents("data")
chunks = split_docs(raw_docs)

print(f"文档总数: {len(raw_docs)} → 分块后数量: {len(chunks)}")

db = FAISS.from_documents(chunks, embedding)
db.save_local("vectorstore/miniLM_faiss")
