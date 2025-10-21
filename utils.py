# from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json, os
from typing import List, Dict, Tuple, Optional

# ============ 工具函数 ============
def render(tok, messages: List[Dict[str, str]]) -> str:
    """按 chat_template 渲染成最终提示词文本（不分词）。"""
    return tok.apply_chat_template(messages, tokenize=False)
    
def _ensure_alternating(messages):
    if not messages:
        return
    if messages[0]["role"] != "user":
        raise ValueError("messages[0] 必须是 'user'（你的模板要求从 user 开始）")
    for i, m in enumerate(messages):
        expect_user = (i % 2 == 0)
        if (m["role"] == "user") != expect_user:
            raise ValueError(f"对话必须严格交替 user/assistant，在索引 {i} 处发现 {m['role']}")

def trim_by_tokens(tok, messages, prompt_budget):
    """
    只保留 messages[0]（persona 的 user）+ 一个“从奇数索引开始的后缀”，
    用二分法找到能放下的最长后缀。这样可保证交替不被破坏。
    """
    if not messages:
        return []

    _ensure_alternating(messages)

    # 只有 persona 这一条时，直接返回
    if len(messages) == 1:
        return messages

    # 允许的后缀起点：奇数索引（index 1,3,5,... 都是 assistant），
    # 这样拼接到 index0(user) 后才能保持交替。
    cand_idx = [k for k in range(1, len(messages)) if k % 2 == 1]

    # 如果任何也放不下，就只留 persona
    best = [messages[0]]

    # 二分：起点越靠前 → 保留消息越多 → token 越大（单调）
    lo, hi = 0, len(cand_idx) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        k = cand_idx[mid]
        candidate = [messages[0]] + messages[k:]
        toks = len(tok(tok.apply_chat_template(candidate, tokenize=False),
                       add_special_tokens=False).input_ids)
        if toks <= prompt_budget:
            best = candidate     # 能放下：尝试保留更多（向左走）
            hi = mid - 1
        else:
            lo = mid + 1         # 放不下：丢更多旧消息（向右走）

    return best

# ============ 原子写 可能会和onedrive同步冲突============
# def atomic_write_json(path: Path, data) -> None:
#     tmp = path.with_suffix(path.suffix + ".tmp")
#     with open(tmp, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
#         f.flush()
#         os.fsync(f.fileno())
#     os.replace(tmp, path)  # 同目录原子替换

# 直接覆盖
def write_json_overwrite(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)    
        
# ============ 存储层 ============
class MsgStore:
    def __init__(self, base_dir: str | Path = "./msgs"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.archive = self.base / "archive.jsonl"  # 只追加
        self.trimmed = self.base / "trimmed.json"   # 当前上下文
        if not self.archive.exists():
            self.archive.write_text("", encoding="utf-8")
        if not self.trimmed.exists():
            self.trimmed.write_text("[]", encoding="utf-8")

    def load_trimmed(self) -> List[Dict[str, str]]:
        try:
            return json.loads(self.trimmed.read_text(encoding="utf-8"))
        except Exception:
            return []

    def save_trimmed(self, messages: List[Dict[str, str]]) -> None:
        write_json_overwrite(self.trimmed, messages)

    def append_archive(self, role: str, content: str, meta: dict | None = None) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "role": role, "content": content}
        if meta: rec["meta"] = meta
        with open(self.archive, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())

# ============ Chat ============
def chat_step(
    user_prompt: str,
    pipe,                     # transformers.pipeline
    tok,                      # AutoTokenizer
    messages: Optional[List[Dict[str, str]]] = None,
    mode: str = "continue",   # "new" | "continue" | "load"
    persona: Optional[str] = None,  # 新开会话时需要，需包含 <<SYS>>…<</SYS>>
    max_context: int = 8192,
    max_new_tokens: int = 256,
    store_dir: str | Path = "./msgs",
    **gen_kwargs,             # 透传生成参数：do_sample/temperature/top_p/repetition_penalty 等
) -> Tuple[str, List[Dict[str, str]], str]:
    """
    运行一轮对话但不保存。
    返回: (reply, messages, user_content_this_turn)
    """
    store = MsgStore(store_dir)

    if mode not in {"new", "continue", "load"}:
        raise ValueError("mode 必须是 'new' | 'continue' | 'load'")

    if mode == "new":
        if not persona:
            raise ValueError("mode='new' 时必须提供 persona（含 <<SYS>>…<</SYS>>）")
        messages = [{"role": "user", "content": f"{persona}\n\n{user_prompt}".strip()}]

    elif mode == "continue":
        if not messages:
            if persona:
                # 没有现成会话但给了 persona，则视作新会话
                messages = [{"role": "user", "content": f"{persona}\n\n{user_prompt}".strip()}]
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
            messages = [{"role": "user", "content": f"{persona}\n\n{user_prompt}".strip()}]
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

# ============ 显式保存（手动调用才落盘） ============
def persist_messages(
    messages: List[Dict[str, str]],
    store_dir: str | Path = "./msgs",
    archive_last_turn: bool = True,
) -> None:
    store = MsgStore(store_dir)
    _ensure_alternating(messages)

    # 1) 覆写 trimmed.json（原子）
    store.save_trimmed(messages)

    # 2) 追加最近一轮到 archive.jsonl（可选）
    if not archive_last_turn:
        return

    # 从尾部向前找最近的一对 (user, assistant)
    pair = None
    for i in range(len(messages) - 2, -1, -1):
        if (
            messages[i]["role"] == "user"
            and i + 1 < len(messages)
            and messages[i + 1]["role"] == "assistant"
        ):
            pair = (messages[i]["content"], messages[i + 1]["content"])
            break

    if pair:
        u, a = pair
        store.append_archive("user", u)
        store.append_archive("assistant", a)
    # 若没有找到成对（比如你在生成前就调用了 persist），就只写 trimmed，不归档
