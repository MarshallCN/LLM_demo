# === 你已有的准备代码（保持不变/已运行） =========================
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import gradio as gr
from gradio.themes.utils import fonts
from utils import chat_step, persist_messages
import uuid
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
for name in ("accelerate", "accelerate.utils", "accelerate.utils.modeling"):
    logging.getLogger(name).setLevel(logging.ERROR)
if gr.NO_RELOAD:
    local_dir = r"C:\Users\c1052689\hug_models\Mistral7B_GPTQ"
    tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(local_dir, device_map="auto", trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.config.pad_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.eos_token_id
    
    pipe = pipeline("text-generation", model=model, tokenizer=tok)

MAX_CONTEXT = 2048 #8192
GEN_BUDGET  = 256
assistant_name = "Nova"; 
user_name = "Marshall"
persona = f"""
<<SYS>>
- Your name is {assistant_name}. Refer to yourself as "{assistant_name}".
- The user's name is {user_name}. Address the user as "{user_name}" when appropriate.
- Use British English and London timezone.
- Do NOT prefix with "Q:" or "A:". Do NOT restate the user's question.
- Output Markdown; code in fenced blocks with a language tag.
- If info is missing, ask at most one clarifying question; otherwise make a reasonable assumption and state it.
- Answer concisely.
<</SYS>>
""".strip()
# ===============================================================

theme = gr.themes.Soft(
    font=[
        fonts.Font("Segoe UI"),
        fonts.Font("system-ui"),
        fonts.Font("sans-serif"),
    ],
    font_mono=[
        fonts.Font("Consolas"),
        fonts.Font("ui-monospace"),
        fonts.Font("monospace"),
    ],
)

css = """
#user_box textarea::-webkit-scrollbar { display: none; }      /* Chrome/Safari */
#user_box textarea { scrollbar-width: none; -ms-overflow-style: none; } /* Firefox/Edge */
"""

GEN_KWARGS = dict(
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.07,
    max_context=MAX_CONTEXT, 
    max_new_tokens=GEN_BUDGET
)

BASE_MSG_DIR = Path("./msgs") 

def mk_msg_dir() -> str:
    m_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    Path(BASE_MSG_DIR, m_id).mkdir(parents=True, exist_ok=True)
    return m_id  # 只返回 ID

def _as_dir(m_id: str) -> str:
    # 统一把传入值规整为 ./msgs/<ID>
    return Path(BASE_MSG_DIR, m_id)

def msg2hist(msg):
    if msg != None:
        chat_history = msg.copy()                 # 外层列表浅拷
        chat_history[0] = msg[0].copy()           # 这个字典单独拷
        chat_history[0]['content'] = chat_history[0]['content'][436:]
        return chat_history

def ui_submit(user_input, messages, msg_id, sessions):
    # 输入 user_input, 消息队列，msg id, sessions list
    user_input = (user_input or "").strip()
    chat_history = msg2hist(messages)
    if not user_input:
        return gr.update(), messages, chat_history, "", msg_dir, gr.update(), 

    # 状态里一律存ID
    msg_id = msg_id if msg_id else ""

    new_session = (not messages)
    # sessions = gr.update()

    if new_session and not msg_id: #刚load界面没有任何msg_id
        msg_id = mk_msg_dir()  # 用户创建msg_id./msgs/<ID>
        sessions = list(sessions or []) + [msg_id]

     #如果有msg_id但是没在sessions 里
    if msg_id and msg_id not in (sessions or []):
        sessions = list(sessions or []) + [msg_id]
    
    sessions_update = gr.update(choices=sessions, value=msg_id)
    
    if new_session:
        reply, messages, mode = chat_step(
            user_input, pipe, tok,
            mode="new", persona=persona,
            **GEN_KWARGS,
        )
    else:
        reply, messages, mode = chat_step(
            user_input, pipe, tok, persona=persona,
            mode="continue", messages=messages,
            **GEN_KWARGS,
        )

    msg_dir = _as_dir(msg_id)
    if msg_dir:
        persist_messages(messages, msg_dir, archive_last_turn=True)

    chat_history = (chat_history or []) + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": reply},
    ]
    return "", messages, chat_history, msg_id, sessions_update, sessions
    
def _load_latest(msg_id: str) -> List[Dict[str, str]]:
    p = Path(_as_dir(msg_id),"trimmed.json")
    if p.exists():
        messages = json.loads(p.read_text(encoding="utf-8"))
        return messages

def _init_sessions():
    sessions = [p.name for p in BASE_MSG_DIR.iterdir() if p.is_dir()]  # 只用ID
    sessions.sort(reverse=True)
    msg_id = sessions[0]
    messages = _load_latest(msg_id)
    chat_history = msg2hist(messages)
    sessions_update = gr.update(choices=sessions, value=msg_id)
    return sessions_update, sessions, msg_id, messages, chat_history

def load_session(session_list, sessions):
    msg_id = session_list   # session_list is the selected msg_id in UI
    messages = _load_latest(msg_id)
    chat_history = msg2hist(messages)
    sessions_update = gr.update(choices=sessions, value=msg_id)
    return msg_id, messages, chat_history, sessions_update

def start_new_session(sessions):
    msg_id = mk_msg_dir()
    sessions = list(sessions or []) + [msg_id]
    sessions_update = gr.update(choices=sessions, value=msg_id)
    return [], [], "", msg_id, sessions_update, sessions  # 返回ID

with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown("## 🗨️ Mistral Chat")

    with gr.Row():
        with gr.Column(scale=3):
            session_list = gr.Radio(choices=[], value=None, label="Conversations", interactive=True)
            new_btn = gr.Button("New session", variant="secondary")

        with gr.Column(scale=9):
            chat = gr.Chatbot(
                label="Chat",
                height=560,
                render_markdown=True,
                type="messages",
                elem_id="chatpane",
            )
            user_box = gr.Textbox(
                label="Your message",
                placeholder="Type and press Enter…",
                autofocus=True,
                elem_id="user_box"
            )
            send = gr.Button("Send", variant="primary")

    messages = gr.State([]) # with persona
    chat_history  = gr.State([])  # without persona
    msg_id = gr.State("")  # 当前会话
    sessions = gr.State([])   # 所有 msg_id list
    
    user_box.submit(
        ui_submit,
        inputs=[user_box, messages, msg_id, sessions],
        outputs=[user_box, messages, chat, msg_id, session_list, sessions],
    )
    send.click(
        ui_submit,
        inputs=[user_box, messages, msg_id, sessions],
        outputs=[user_box, messages, chat, msg_id, session_list, sessions],
    )
    new_btn.click(
        start_new_session,
        inputs=[sessions],
        outputs=[messages, chat, user_box, msg_id, session_list, sessions],
    )
    session_list.change(load_session,
                        inputs=[session_list, sessions], # session_list is msg_id here
                        outputs=[msg_id, messages, chat, session_list]
    )

    demo.load(_init_sessions, None,
              outputs=[session_list, sessions, msg_id, messages, chat])
demo.launch(debug=True)
