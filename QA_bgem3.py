# 假设向量数据库已经搭建好，加载Qwen2.5-0.5B-Instruct，内存里量化成4-bit, 默认使用base 不添加微调的adapter
# from langchain_community.document_loaders import TextLoader, PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings 
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
import os
import pickle
import faiss, pickle, numpy as np
from FlagEmbedding import BGEM3FlagModel

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# from peft import PeftModel

BASE_DIR = r"C:\Users\c1052689\hug_models\Qwen2.5-0.5B-Instruct"
ADAPTER  = r".\qwen0.5b-mini-platypus-qlora"

tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=False, local_files_only=True)
# bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
#                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
# base = AutoModelForCausalLM.from_pretrained(BASE_DIR, quantization_config=bnb, device_map="auto", local_files_only=True)
base = AutoModelForCausalLM.from_pretrained(BASE_DIR, device_map="auto", local_files_only=True) # With no Q
# model = PeftModel.from_pretrained(base, ADAPTER) # QLora mode;

# 1. 加载语料 & 索引
with open("vectorstore/corpus.pkl", "rb") as f:
    corpus = pickle.load(f)
index = faiss.read_index("vectorstore/bgem3.index")

# 2. 加载Embedding模型
BGEM3 = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
reranker = CrossEncoder("BAAI/bge-reranker-large")

pipe = pipeline("text-generation", model=base, tokenizer=tok, max_new_tokens=512)

def build_prompt_corpus(corpus, question):
    context_text = "\n\n".join(corpus)
    
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
    # top_docs = db.similarity_search(query, k=3) #RAG-baseline
    qv = np.array(BGEM3.encode([query])["dense_vecs"], dtype="float32")
    faiss.normalize_L2(qv)
    D, I = index.search(qv, 8)
    results = [corpus[i] for i in I[0]] #粗筛
    pairs = [[query, c] for c in results]
    scores = reranker.predict(pairs) #算分细排
    top_docs = [c for _, c in sorted(zip(scores, results), reverse=True)][:3]

    prompt = build_prompt_corpus(top_docs, query)
    out = pipe(
        prompt,
        max_new_tokens=1024,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        eos_token_id=tok.eos_token_id,     
        pad_token_id=tok.eos_token_id,     # Qwen 常用 eos 作为 pad
        return_full_text=False,            # 只要新增内容，不要把 prompt 也返回
    )
    reply = out[0]["generated_text"]
    
    print("\n🧠 Answer: ")
    print(reply)

    print("\n📄 Reference Context:")
    for i, doc in enumerate(top_docs):
        print(f"[{i+1}] {doc[:200]}...\n")