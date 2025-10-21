# === 你已有的准备代码（保持不变/已运行） =========================
# from __future__ import annotations
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
local_dir = r"C:\Users\c1052689\hug_models\Mistral7B_GPTQ"
tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(local_dir, device_map="auto", trust_remote_code=True)
tok.pad_token = tok.eos_token
tok.padding_side = "left"
model.config.pad_token_id = tok.eos_token_id
model.generation_config.pad_token_id = tok.eos_token_id

import logging
for name in ("accelerate", "accelerate.utils", "accelerate.utils.modeling"):
    logging.getLogger(name).setLevel(logging.ERROR)

import gradio as gr
from gradio.themes.utils import fonts
from utils import chat_step, persist_messages
import uuid
import glob
from datetime import datetime
from pathlib import Path
from datetime import datetime, timezone
import json
from typing import List, Dict, Tuple, Optional

pipe = pipeline("text-generation", model=model, tokenizer=tok)
MAX_CONTEXT = 8192
GEN_BUDGET  = 256
assistant_name = "Nova"
user_name = "Marshall"
persona = f"""
<<SYS>>
- Your name is {assistant_name}. Refer to yourself as "{assistant_name}".
- The user's name is {user_name}. Address the user as "{user_name}" when appropriate.
- Use British English and London timezone.
- Do NOT prefix with "Q:" or "A:". Do NOT restate the user's question.
- Output Markdown; code in fenced blocks with a language tag.
- If info is missing, ask at most one clarifying question; otherwise make a reasonable assumption and state it.
<</SYS>>
""".strip()

GEN_KWARGS  = dict(
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.05,
)

# ====== 多会话持久化 ======

BASE_MSG_DIR = "./msgs"
Path(BASE_MSG_DIR).mkdir(parents=True, exist_ok=True)

def _make_m_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]

def _ensure_dir_for(m_id: str) -> None:
    Path(BASE_MSG_DIR, m_id).mkdir(parents=True, exist_ok=True)

def _load_latest(m_id: str) -> List[Dict[str, str]]:
    """
    只加载 ./msgs/{m_id}/trimmed.json（由persist_messages 写入）。
    若不存在则返回空列表。
    """
    p = Path(BASE_MSG_DIR, m_id, "trimmed.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _as_chat_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # 仅把 user/assistant 展示在 Chatbot（system/persona 不展示）
    return [m for m in (messages or []) if m.get("role") in ("user", "assistant")]

# ==== 交互逻辑 ====
def ui_submit(user_input, current_m_id, messages_state, chat_history):
    user_input = (user_input or "").strip()
    if not user_input:
        return gr.update(), messages_state, chat_history, ""

    # 没有会话ID则新建
    if not current_m_id:
        current_m_id = _make_m_id()
        _ensure_dir_for(current_m_id)

    # 是否新会话：以 messages_state 是否为空为准
    mode = "new" if not messages_state else "continue"

    # === 调用你自己的 chat_step（保持原有签名）===
    reply, messages_state, _ = chat_step(
        user_input, pipe, tok,
        mode=mode,
        persona=persona,
        max_context=MAX_CONTEXT,
        max_new_tokens=GEN_BUDGET,
        **GEN_KWARGS
    )

    # === 立刻落盘 ===
    persist_messages(messages_state, f"{BASE_MSG_DIR}/{current_m_id}", archive_last_turn=True)

    # 刷新展示
    chat_history = _as_chat_messages(messages_state)
    return chat_history, messages_state, chat_history, ""  # 清空输入框

def start_new_session(session_ids):
    """
    新建一个会话：生成 m_id，加入列表并切到该会话；清空消息。
    """
    m_id = _make_m_id()
    _ensure_dir_for(m_id)
    new_ids = list(session_ids or []) + [m_id]
    return (
        gr.update(choices=new_ids, value=m_id),  # session_list
        new_ids,                                 # session_ids_state
        m_id,                                    # current_m_id
        [], [], [], ""                           # messages_state, chat_history, chat, user_box
    )

def load_session(selected_m_id):
    """
    切换会话：只读 trimmed.json 并回显历史记录。
    """
    msgs = _load_latest(selected_m_id) or []
    # === 仅用于“显示”的视图，跳过前2条 ===
    msgs_for_view = msgs[1:] if len(msgs) > 1 else []
    chat = _as_chat_messages(msgs_for_view)
    # 注意：messages_state 仍然保留完整 msgs（包括前两条），模型需要完整上下文
    return selected_m_id, msgs, chat, chat

# ==== Gradio UI ====
theme = gr.themes.Soft()
css = ""

with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown("## 🧠 Local Chat (Multi-session + Persistence)")

    with gr.Row():
        with gr.Column(scale=3):
            session_list = gr.Radio(choices=[], value=None, label="Conversations")
            new_btn = gr.Button("New session", variant="secondary")
        with gr.Column(scale=9):
            chat = gr.Chatbot(
                label="Chat",
                height=560,
                render_markdown=True,
                type="messages",
            )
            user_box = gr.Textbox(
                label="Your message",
                placeholder="Type and press Enter…",
                autofocus=True,
            )
            send = gr.Button("Send", variant="primary")

    # 状态
    session_ids_state = gr.State([])   # 所有 m_id
    current_m_id      = gr.State(None) # 当前会话
    messages_state    = gr.State([])   # 完整 messages（含 system/persona）
    chat_history      = gr.State([])   # 展示用（仅 user/assistant）

    # 绑定
    user_box.submit(ui_submit,
                    inputs=[user_box, current_m_id, messages_state, chat_history],
                    outputs=[chat, messages_state, chat_history, user_box])
    send.click(ui_submit,
               inputs=[user_box, current_m_id, messages_state, chat_history],
               outputs=[chat, messages_state, chat_history, user_box])

    new_btn.click(start_new_session,
                  inputs=[session_ids_state],
                  outputs=[session_list, session_ids_state, current_m_id, messages_state, chat_history, chat, user_box])

    session_list.change(load_session,
                        inputs=[session_list],
                        outputs=[current_m_id, messages_state, chat_history, chat])

    # 启动时扫描 ./msgs，填充会话并默认加载最新一个（按目录名时间倒序）
    def _init_sessions():
        Path(BASE_MSG_DIR).mkdir(parents=True, exist_ok=True)
        ids = [d for d in os.listdir(BASE_MSG_DIR) if (Path(BASE_MSG_DIR, d).is_dir())]
        ids.sort(reverse=True)
        if not ids:
            m_id = _make_m_id(); _ensure_dir_for(m_id); ids = [m_id]
            return (gr.update(choices=ids, value=m_id), ids, m_id, [], [], [])
        m_id = ids[0]
        msgs = _load_latest(m_id)
        # === 初次加载也同样跳过前2条只用于显示 ===
        msgs_for_view = msgs[2:] if len(msgs) > 2 else []
        chat_msgs = _as_chat_messages(msgs_for_view)
        return (gr.update(choices=ids, value=m_id), ids, m_id, msgs, chat_msgs, chat_msgs)

    demo.load(_init_sessions, None,
              outputs=[session_list, session_ids_state, current_m_id, messages_state, chat_history, chat])

if __name__ == "__main__":
    demo.launch()