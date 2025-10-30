# Build vector DB with bge-m3 embedding
# pip install FlagEmbedding sentence-transformers faiss-cpu  # 需 Python>=3.9
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
import faiss, numpy as np
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

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

raw_docs = load_documents("data")
chunks = split_docs(raw_docs)
corpus = [t.page_content for t in chunks]

# 1) 编码语料并建索引
BGEM3 = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)    # 多语 & dense+lexical
dense = BGEM3.encode(corpus, batch_size=64)["dense_vecs"]
# 保证是 numpy array 且 float32 类型
if not isinstance(dense, np.ndarray):
    dense = np.array(dense)
dense = dense.astype('float32')
faiss.normalize_L2(dense)
index = faiss.IndexFlatIP(dense.shape[1])
index.add(dense)

# 保存向量数据库
faiss.write_index(index, "vectorstore/bgem3.index")
# 保存语料
with open("vectorstore/corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)
    
# # 2) 粗检索 Top-k=8
# q = "metapath"
# qv = BGEM3.encode([q])["dense_vecs"]; faiss.normalize_L2(qv)
# D, I = index.search(qv, 8)
# cands = [corpus[i] for i in I[0]]

# # 3) 交叉编码器重排后取 Top-3
# reranker = CrossEncoder("BAAI/bge-reranker-large")
# pairs = [[q, c] for c in cands]
# scores = reranker.predict(pairs)
# top3 = [c for _, c in sorted(zip(scores, cands), reverse=True)][:3]
# print(top3)