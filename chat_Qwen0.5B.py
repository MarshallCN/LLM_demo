# === Qwen 2.5 Coder 1.5B or 0.5B GTQ_Int4 =========================
# 增加参数选择，删除
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging
for name in ("accelerate", "accelerate.utils", "accelerate.utils.modeling"):
    logging.getLogger(name).setLevel(logging.ERROR)
    
import stat, shutil
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
    # local_dir = r"C:\Users\c1052689\hug_models\Qwen2.5-0.5B-Instruct" # full model no Q
    tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(local_dir, device_map="auto", trust_remote_code=True) # ,torch_dtype=torch.bfloat16
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.config.pad_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.eos_token_id

    pipe = pipeline("text-generation", model=model, tokenizer=tok)

# ===================== Defaults =====================
MAX_CONTEXT = 2048  # prompt budget + generation
GEN_BUDGET  = 256
assistant_name = "Nova"
user_name = "Marshall"
# 单行默认 persona（节省 token）
DEFAULT_SYS_PROMPT = (
    f"Your name is {assistant_name}; address the user as {user_name} when appropriate; no Q:/A: prefixes; use Markdown; fence code with a language tag; Be concise but substantive."
)

# 仅作为初始值；真正的生成参数来自 UI 的滑块
GEN_KWARGS = dict(
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1.05,
    max_context=MAX_CONTEXT,
    max_new_tokens=GEN_BUDGET,
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

# ============ Chat core ============

# def _ensure_system(messages: Optional[List[Dict[str, str]]], sys_prompt: str) -> List[Dict[str, str]]:
#     """确保第一个消息是系统提示，并与当前文本框一致。"""
#     sys_prompt = (sys_prompt or DEFAULT_SYS_PROMPT).strip()
#     if not messages:
#         return [{"role": "system", "content": sys_prompt}]
#     messages = list(messages)
#     if messages[0].get("role") != "system":
#         messages.insert(0, {"role": "system", "content": sys_prompt})
#     else:
#         messages[0] = {"role": "system", "content": sys_prompt}
#     return messages


def chat_step(
    user_prompt: str,
    pipe,                     # transformers.pipeline
    tok,                      # AutoTokenizer
    messages: Optional[List[Dict[str, str]]] = None,
    mode: str = "continue",   # "new" | "continue" | "load"
    persona: Optional[str] = None,  # 兼容旧参数；会被 sys_prompt 覆盖
    max_context: int = 8192,
    max_new_tokens: int = 256,
    **gen_kwargs,             # do_sample/temperature/top_p/repetition_penalty 等
) -> Tuple[str, List[Dict[str, str]], str]:
    """
    运行一轮对话但不保存。
    返回: (reply, messages, mode)
    """
    if mode not in {"new", "continue", "load"}:
        raise ValueError("mode 必须是 'new' | 'continue' | 'load'")

    # 将用户输入加入到 messages
    if mode == "new":
        if not persona:
            persona = DEFAULT_SYS_PROMPT
        messages = [{"role": "system", "content": persona}, {"role": "user", "content": user_prompt.strip()}]
    elif mode in {"continue", "load"}:
        if not messages:
            messages = [{"role": "system", "content": persona or DEFAULT_SYS_PROMPT}]
            print('not mesaage')
        messages.append({"role": "user", "content": user_prompt})
        print('append')
    print(messages)

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

# ============ UI callbacks ============

def ui_submit(user_input, messages, msg_id, sessions, sys_prompt,
              temperature, top_p, max_new_tokens, repetition_penalty):
    user_input = (user_input or "").strip()
    # 可视化历史不显示 system（用旧工具函数）
    chat_history = msg2hist(sys_prompt, messages)
    if not user_input:
        sessions_update = gr.update(choices=sessions, value=(msg_id or None))
        return gr.update(), messages, chat_history, msg_id, sessions_update, sessions

    # 统一维护 system
    # messages = _ensure_system(messages, sys_prompt)

    # 会话 ID 维护
    msg_id = msg_id if msg_id else ""
    new_session = (len(messages) <= 1)  # 只有 system 视为新会话
    if new_session and not msg_id:
        msg_id = mk_msg_dir(BASE_MSG_DIR)
        sessions = list(sessions or []) + [msg_id]
    if msg_id and msg_id not in (sessions or []):
        sessions = list(sessions or []) + [msg_id]
    sessions_update = gr.update(choices=sessions, value=msg_id)

    # 动态生成参数（来自 UI）
    gen_cfg = dict(
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
    )

    reply, messages, mode = chat_step(
        user_input, pipe, tok,
        mode="continue" if not new_session else "new", 
        messages=messages,
        persona=sys_prompt,
        max_context=MAX_CONTEXT,
        max_new_tokens=int(max_new_tokens),
        **gen_cfg,
    )
    print(mode)
    if len(messages) > 0:
        msg_dir = _as_dir(BASE_MSG_DIR, msg_id)
        persist_messages(messages, msg_dir, archive_last_turn=True)

    chat_history = (chat_history or []) + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": reply},
    ]
    return "", messages, chat_history, msg_id, sessions_update, sessions


def _load_latest(msg_id: str) -> List[Dict[str, str]]:
    p = Path(_as_dir(BASE_MSG_DIR, msg_id), "trimmed.json")
    if p.exists():
        messages = json.loads(p.read_text(encoding="utf-8"))
        return messages
    return []


def _init_sessions():
    sessions = [p.name for p in BASE_MSG_DIR.iterdir() if p.is_dir()] if BASE_MSG_DIR.exists() else []
    if len(sessions) == 0:
        # 无历史会话
        return gr.update(choices=[], value=None), [], "", [], []
    else:
        sessions.sort(reverse=True)
        msg_id = sessions[0]
        messages = _load_latest(msg_id)
        chat_history = msg2hist(DEFAULT_SYS_PROMPT, messages)
        sessions_update = gr.update(choices=sessions, value=msg_id)
        return sessions_update, sessions, msg_id, messages, chat_history

def load_session(session_list, sessions):
    msg_id = session_list
    messages = _load_latest(msg_id)
    chat_history = msg2hist(DEFAULT_SYS_PROMPT, messages)
    sessions_update = gr.update(choices=sessions, value=msg_id)
    return msg_id, messages, chat_history, sessions_update

def start_new_session(sessions):
    msg_id = mk_msg_dir(BASE_MSG_DIR)
    sessions = list(sessions or []) + [msg_id]
    sessions_update = gr.update(choices=sessions, value=msg_id)
    return [], [], "", msg_id, sessions_update, sessions

def export_messages_to_json(messages, msg_id):
    base = Path("/data/exports") if Path("/data").exists() else Path("./exports")
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-")
    fname = f"msgs_{stamp}.json"
    path = base / fname
    path.write_text(json.dumps(messages or [], ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)  # 返回给 gr.File 的文件路径

def on_click_download(messages, msg_id):
    path = export_messages_to_json(messages, msg_id)
    # 让隐藏的 gr.File 出现，并指向刚刚写出的文件
    return gr.update(value=path, visible=True)

def delete_session(msg_id, sessions):
    """删除当前会话目录并刷新会话列表。"""
    if msg_id:
        path = _as_dir(BASE_MSG_DIR, msg_id)
        if path.exists():
            def _onerror(func, p, exc_info):
                 os.chmod(p, stat.S_IWRITE)  # 清只读
            shutil.rmtree(path, onerror=_onerror)  # 只调一次
    return _init_sessions()

# ===================== UI =====================

with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown("## 🧠 Qwen Chat")

    with gr.Row():
        with gr.Column(scale=3):
            # —— 来自 old_gguf：System prompt + Generation settings ——
            sys_prompt = gr.Textbox(
                label="System prompt",
                value=DEFAULT_SYS_PROMPT,
                lines=4,
                show_label=True,
            )
            with gr.Accordion("Generation settings", open=False):
                temperature = gr.Slider(0.0, 2.0, value=GEN_KWARGS["temperature"], step=0.05, label="temperature")
                top_p = gr.Slider(0.1, 1.0, value=GEN_KWARGS["top_p"], step=0.01, label="top_p")
                max_new_tokens = gr.Slider(16, 1024, value=GEN_BUDGET, step=16, label="max_new_tokens")
                repetition_penalty = gr.Slider(1.0, 2.0, value=GEN_KWARGS["repetition_penalty"], step=0.01, label="repetition_penalty")

            session_list = gr.Radio(choices=[], value=None, label="Conversations", interactive=True)
            new_btn = gr.Button("New session", variant="secondary")
            del_btn = gr.Button("Delete session", variant="stop")
            dl_btn = gr.Button("Download JSON", variant="secondary")
            dl_file = gr.File(label="", interactive=False, visible=False, elem_id="dl-file")
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

    messages = gr.State([])  # with system
    chat_history  = gr.State([])  # without system
    msg_id = gr.State("")
    sessions = gr.State([])

    user_box.submit(
        ui_submit,
        inputs=[user_box, messages, msg_id, sessions, sys_prompt,
                temperature, top_p, max_new_tokens, repetition_penalty],
        outputs=[user_box, messages, chat, msg_id, session_list, sessions],
    )
    send.click(
        ui_submit,
        inputs=[user_box, messages, msg_id, sessions, sys_prompt,
                temperature, top_p, max_new_tokens, repetition_penalty],
        outputs=[user_box, messages, chat, msg_id, session_list, sessions],
    )
    new_btn.click(
        start_new_session,
        inputs=[sessions],
        outputs=[messages, chat, user_box, msg_id, session_list, sessions],
    )
    session_list.change(
        load_session,
        inputs=[session_list, sessions],
        outputs=[msg_id, messages, chat, session_list]
    )
    dl_btn.click(
        on_click_download,
        inputs=[messages, msg_id],
        outputs=[dl_file],
    )
    del_btn.click(
        delete_session,
        inputs=[msg_id, sessions],
        outputs=[session_list, sessions, msg_id, messages, chat],
    )
    demo.load(_init_sessions, None,
              outputs=[session_list, sessions, msg_id, messages, chat])

if __name__ == "__main__":
    demo.launch()
