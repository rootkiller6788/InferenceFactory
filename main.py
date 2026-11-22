"""入口：`python main.py <cmd>`。

⚠️ 这里 `sys.path` 指向**本项目**的 `src/`，不靠安装 —— 于是
「从任意 cwd 调用」与「在项目根调用」走的是同一份代码。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

from inference.cli import main  # noqa: E402
from inference.console import use_utf8_console  # noqa: E402

# ⚠️ 必须在 argparse 打印任何 `--help` 之前重配 —— 帮助文本里有 GBK 编不出的符号。
use_utf8_console()

if __name__ == "__main__":
    sys.exit(main())
