"""把 `config.DEFAULTS` 重新导出到 `configs/default.json` 并自检。

⚠️ 为什么不直接用 `main.py dump-config`：那条命令会**先读磁盘配置**，
而磁盘配置里若已有「代码里不存在的键」，保险丝 `no_stale_config_keys`
会在那一步就抛错 ⇒ 鸡生蛋。改 `DEFAULTS` 之后**必须**走这条旁路。

自检：`configs/default.json` 的内容 == `DEFAULTS` 的落盘形式。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.config import DEFAULTS, dump_config, load_config  # noqa: E402
from inference.console import use_utf8_console  # noqa: E402

use_utf8_console()

target = ROOT / "configs" / "default.json"
dump_config(DEFAULTS, target)
back = load_config(target, root=ROOT)
if back != DEFAULTS:
    print("❌ 导出后读回来与 DEFAULTS 不同")
    sys.exit(1)
print(f"✅ 已导出 {target}（{target.stat().st_size} 字节，与 DEFAULTS 一致）")
