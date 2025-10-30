# RAG baseline using miniLM embedding for vector database
# from langchain_community.document_loaders import TextLoader, PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings 
# from pathlib import Path
# # === 文档加载 ===
# def load_documents(folder: str):
#     docs = []
#     for path in Path(folder).rglob("*"):
#         if path.suffix == ".txt":
#             docs += TextLoader(str(path), encoding="utf-8").load()
#         elif path.suffix == ".pdf":
#             docs += PyPDFLoader(str(path)).load()
#     return docs

# # === 文本分块 ===
# def split_docs(docs, chunk_size=512, chunk_overlap=64):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return splitter.split_documents(docs)

# # === 向量化模型 ===
# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # === 主流程 ===
# raw_docs = load_documents("data")
# chunks = split_docs(raw_docs)
# print(f"文档总数: {len(raw_docs)} → 分块后数量: {len(chunks)}")
# db = FAISS.from_documents(chunks, embedding)
# db.save_local("vectorstore/miniLM_faiss")

# ======================== QA part ========================================
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
# === 加载向量库 ===
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore/miniLM_faiss", embedding, allow_dangerous_deserialization=True)

# === 加载本地大模型 ===
BASE_DIR = r"C:\Users\c1052689\hug_models\Qwen2.5-0.5B-Instruct"
ADAPTER  = r".\qwen0.5b-mini-platypus-qlora"

tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=False, local_files_only=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained(BASE_DIR, quantization_config=bnb, device_map="auto", local_files_only=True)
# model = PeftModel.from_pretrained(base, ADAPTER) 
pipe = pipeline("text-generation", model=base, tokenizer=tok, max_new_tokens=512)

# === 拼接 prompt 函数 ===
def build_prompt(context_chunks, question):
    context_text = "\n\n".join([doc.page_content for doc in context_chunks])
    
    user_prompt = f"""Answer the question based on the following context:

{context_text}

Question: {question}
Answer:"""

    full_prompt = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return full_prompt


# === 主循环 ===
while True:
    query = input("\nEnter your question (press q to quit): ")
    if query.strip().lower() in ["q", "quit", "exit"]:
        break

    top_docs = db.similarity_search(query, k=3)
    prompt = build_prompt(top_docs, query)
    out = pipe(
        prompt,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tok.eos_token_id,     # 防止跑飞
        pad_token_id=tok.eos_token_id,     # Qwen 常用 eos 作为 pad
        return_full_text=False,            # 只要新增内容，不要把 prompt 也返回
    )
    reply = out[0]["generated_text"]
    
    print("\n🧠 Answer: ")
    print(reply)

    print("\n📄 Reference Context: ")
    for i, doc in enumerate(top_docs):
        print(f"[{i+1}] {doc.page_content[:200]}...\n")
