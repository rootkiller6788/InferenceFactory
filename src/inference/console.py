"""控制台输出编码：**统一 UTF-8**，否则 Windows 管道下会崩。

⚠️ Windows 上 stdout 是控制台时走 PEP 528（本来就是 UTF-8），但一旦被
**重定向到管道 / 文件**（`subprocess`、pytest、`diagnostics/*.py`），
Python 会用**区域编码**（简体中文环境是 GBK）。本项目的大量文案里有
`⚠️` / `✅` / `❌` 这类 GBK 编不出来的字符 —— 症状是 `python main.py --help`
直接 `UnicodeEncodeError` 退出 1，而输出去向不同时行为还不一样。

⇒ 入口处统一重配成 UTF-8（`errors="replace"` 兜底，绝不再因为一个符号崩）。
文件本来就用 UTF-8 写（见 `emit.write_product` 的 `newline="\\n"`），
控制台跟着统一，上下游读到的才是同一份东西。
"""

from __future__ import annotations

import sys


def use_utf8_console() -> None:
    """把 stdout / stderr 重配成 UTF-8（无法重配的对象静默跳过）。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # 流已关闭 / 不是 TextIOWrapper ⇒ 保持原样，不阻断启动。
            pass
