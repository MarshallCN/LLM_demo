# === Qwen 2.5 Coder 1.5B or 0.5B =========================
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging
for name in ("accelerate", "accelerate.utils", "accelerate.utils.modeling"):
    logging.getLogger(name).setLevel(logging.ERROR)
    
import gradio as gr
from gradio.themes.utils import fonts
import uuid
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional
from utils import render, trim_by_tokens, mk_msg_dir, _as_dir, msg2hist, persist_messages

# ===================== Model (optional local load) =====================
if gr.NO_RELOAD:
    # local_dir = r"C:\Users\c1052689\hug_models\Qwen2.5Coder1_5B_Instruct"
    local_dir = r"C:\Users\c1052689\hug_models\Qwen2.5_0.5B_Instruct_GPTQ_Int4"
    tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(local_dir, device_map="auto", trust_remote_code=True) # ,torch_dtype=torch.bfloat16
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.config.pad_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.eos_token_id

    pipe = pipeline("text-generation", model=model, tokenizer=tok)

# ===================== Defaults =====================
MAX_CONTEXT = 2048 #8192
GEN_BUDGET  = 256
assistant_name = "Nova"; 
user_name = "Marshall"
persona = f"""Your name is {assistant_name}. Address the user as "{user_name}" when appropriate. Do not add "Q:"/"A:" prefixes. Output Markdown; code in fenced blocks with a language tag. Be concise but never give empty feedback.
""".strip()

GEN_KWARGS = dict(
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1.05,
    max_context=MAX_CONTEXT, 
    max_new_tokens=GEN_BUDGET
)

BASE_MSG_DIR = Path("./msgs/msgs_Qwen") 

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
# ============ Chat ============
def chat_step(
    user_prompt: str,
    pipe,                     # transformers.pipeline
    tok,                      # AutoTokenizer
    messages: Optional[List[Dict[str, str]]] = None,
    mode: str = "continue",   # "new" | "continue" | "load"
    persona: Optional[str] = None,  # 新开会话时需要，
    max_context: int = 8192,
    max_new_tokens: int = 256,
    store_dir: str | Path = "./msgs",
    **gen_kwargs,             # 透传生成参数：do_sample/temperature/top_p/repetition_penalty 等
) -> Tuple[str, List[Dict[str, str]], str]:
    """
    运行一轮对话但不保存。
    返回: (reply, messages, user_content_this_turn)
    """
    if mode not in {"new", "continue", "load"}:
        raise ValueError("mode 必须是 'new' | 'continue' | 'load'")

    if mode == "new":
        if not persona:
            raise ValueError("mode='new' 时必须提供 persona")
        messages = [{"role": "system", "content": persona}, {"role": "user", "content": user_prompt.strip()}]

    elif mode == "continue":
        if not messages:
            if persona:
                # 没有现成会话但给了 persona，则视作新会话
                messages = [{"role": "system", "content": persona}, {"role": "user", "content": user_prompt.strip()}]
                mode = "new"
            else:
                raise ValueError("mode='continue' 需要传入非空 messages，或改用 mode='new' 并提供 persona")
        else:
            messages.append({"role": "user", "content": user_prompt})

    elif mode == "load":
        messages = store.load_trimmed()
        if not messages:
            if not persona:
                raise ValueError("磁盘没有可加载的会话，且未提供 persona 以新建。")
            messages = [{"role": "system", "content": persona}, {"role": "user", "content": user_prompt.strip()}]
            mode = "new"   # 实际上是新开
        else:
            messages.append({"role": "user", "content": user_prompt})

    # 裁剪 → 渲染 → 生成
    prompt_budget = max_context - max_new_tokens
    messages = trim_by_tokens(tok, messages, prompt_budget)
    text = render(tok, messages)
    out = pipe(
        text,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
        **gen_kwargs,
    )
    reply = out[0]["generated_text"].strip()

    # 追加 assistant，二次裁剪
    messages.append({"role": "assistant", "content": reply})
    messages = trim_by_tokens(tok, messages, prompt_budget)
    return reply, messages, mode

# ============ UI ============

def ui_submit(user_input, messages, msg_id, sessions):
    # 输入 user_input, 消息队列，msg id, sessions list
    user_input = (user_input or "").strip()
    chat_history = msg2hist(persona, messages)
    if not user_input:
        return gr.update(), messages, chat_history, "", msg_dir, gr.update(), 

    # 状态里一律存ID
    msg_id = msg_id if msg_id else ""

    new_session = (not messages)
    # sessions = gr.update()

    if new_session and not msg_id: #刚load界面没有任何msg_id
        msg_id = mk_msg_dir(BASE_MSG_DIR)  # 用户创建msg_id./msgs/<ID>
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

    if len(messages)>0:
        msg_dir = _as_dir(BASE_MSG_DIR, msg_id)
        persist_messages(messages, msg_dir, archive_last_turn=True)

    chat_history = (chat_history or []) + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": reply},
    ]
    return "", messages, chat_history, msg_id, sessions_update, sessions
    
def _load_latest(msg_id: str) -> List[Dict[str, str]]:
    p = Path(_as_dir(BASE_MSG_DIR, msg_id),"trimmed.json")
    if p.exists():
        messages = json.loads(p.read_text(encoding="utf-8"))
        return messages

def _init_sessions():
    sessions = [p.name for p in BASE_MSG_DIR.iterdir() if p.is_dir()]  # 只用ID
    if len(sessions)==0:
        return gr.update(choices=[], value=None), [], "", [], []
    else:
        sessions.sort(reverse=True)
        msg_id = sessions[0]
        messages = _load_latest(msg_id)
        chat_history = msg2hist(persona, messages)
        sessions_update = gr.update(choices=sessions, value=msg_id)
        return sessions_update, sessions, msg_id, messages, chat_history
            

def load_session(session_list, sessions):
    msg_id = session_list   # session_list is the selected msg_id in UI
    messages = _load_latest(msg_id)
    chat_history = msg2hist(persona, messages)
    sessions_update = gr.update(choices=sessions, value=msg_id)
    return msg_id, messages, chat_history, sessions_update

def start_new_session(sessions):
    msg_id = mk_msg_dir(BASE_MSG_DIR)
    sessions = list(sessions or []) + [msg_id]
    sessions_update = gr.update(choices=sessions, value=msg_id)
    return [], [], "", msg_id, sessions_update, sessions  # 返回ID

with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown("## 🧠 Qwen Chat")

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
demo.launch()
