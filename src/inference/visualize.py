"""从 `out/` **重建**报告并逐位比对。

这一条只读命令买到的是比「跑一遍看报告长什么样」更强的命题：
**产物足以重造报告**。也就是说报告里没有任何一句话是
「只有运行时的内存里有、盘上没有」的东西。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import emit as E
from .report import render


def rebuild(out_dir: str | Path) -> tuple[dict[str, Any], str, str]:
    products = E.read_products(Path(out_dir))
    md, html = render(products)
    return products, md, html


def verify(out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    products, md, html = rebuild(out)
    rows = []
    for name, text in (("report.md", md), ("report.html", html)):
        p = out / name
        if not p.exists():
            rows.append({"file": name, "exists": False, "bitwise_equal": False,
                         "bytes": len(text.encode("utf-8")),
                         "chars": len(text)})
            continue
        disk = p.read_text(encoding="utf-8")
        blob = text.encode("utf-8")
        rows.append({
            "file": name, "exists": True,
            "bitwise_equal": disk == text,
            # ⚠️ `bytes` 是**真的字节数**：`len(str)` 是字符数，中文下差 1.3 倍。
            #    两个都报，免得「对不上」的时候不知道是差在哪一层。
            "bytes": len(blob),
            "chars": len(text),
        })
    return {"out_dir": str(out), "rows": rows,
            "all_equal": all(r["bitwise_equal"] for r in rows)}
