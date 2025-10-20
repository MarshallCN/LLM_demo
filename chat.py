# === 你已有的准备代码（保持不变/已运行） =========================
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

from utils import chat_step, persist_messages
pipe = pipeline("text-generation", model=model, tokenizer=tok)

MAX_CONTEXT = 8192
GEN_BUDGET  = 256
assistant_name = "Nova"; user_name = "Marshall"
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
# ===============================================================

import gradio as gr

GEN_KWARGS = dict(
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.07,
)

def ui_submit(user_input, mode, messages_state, chat_history):
    user_input = (user_input or "").strip()
    if not user_input:
        return gr.update(), messages_state, chat_history, ""

    new_session = (mode == "new") or (not messages_state)

    if new_session:
        reply, messages_state, _ = chat_step(
            user_input, pipe, tok,
            mode="new", persona=persona,
            max_context=MAX_CONTEXT, max_new_tokens=GEN_BUDGET,
            **GEN_KWARGS,
        )
    else:
        reply, messages_state, _ = chat_step(
            user_input, pipe, tok,
            mode="continue", messages=messages_state,
            max_context=MAX_CONTEXT, max_new_tokens=GEN_BUDGET,
            **GEN_KWARGS,
        )

    # Chatbot 历史：OpenAI 风格的 role/content 列表
    chat_history = (chat_history or []) + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": reply},
    ]
    return chat_history, messages_state, chat_history, ""  # 清空输入框

def start_new_session():
    # 清空对话并切到 new
    return [], [], "new", ""
    
from gradio.themes.utils import fonts
theme = gr.themes.Soft(
    font=[
        fonts.Font("Microsoft YaHei UI"),   # 你机器上的系统字体
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
    .wrap.svelte-1cl84cg {max-width: 900px; margin: 0 auto;}
"""
with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown("## 🗨️ Local Mistral Chat")

    with gr.Row():
        with gr.Column(scale=3):
            mode_radio = gr.Radio(
                choices=["continue", "new"],
                value="continue",
                label="Mode",
                info="",
            )
            new_btn = gr.Button("New session", variant="secondary")

        with gr.Column(scale=9):
            chat = gr.Chatbot(
                label="Chat",
                height=560,
                render_markdown=True,
                type="messages",
                elem_id="chatpane",   # ✅ 新增：给前端脚本一个可定位的容器
            )
            user_box = gr.Textbox(
                label="Your message",
                placeholder="Type and press Enter…",
                autofocus=True,
            )
            send = gr.Button("Send", variant="primary")

    # 状态：模型使用的 messages（role/content），以及 Chatbot 的 messages 历史
    messages_state = gr.State([])  # 给 chat_step 用
    chat_history  = gr.State([])   # 给 Chatbot 展示用（role/content 列表）

    user_box.submit(
        ui_submit,
        inputs=[user_box, mode_radio, messages_state, chat_history],
        outputs=[chat, messages_state, chat_history, user_box],
    )
    send.click(
        ui_submit,
        inputs=[user_box, mode_radio, messages_state, chat_history],
        outputs=[chat, messages_state, chat_history, user_box],
    )
    new_btn.click(
        start_new_session,
        inputs=None,
        outputs=[chat, messages_state, mode_radio, user_box],
    )

demo.launch()
