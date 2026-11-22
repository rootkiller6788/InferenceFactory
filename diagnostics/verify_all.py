"""最终验收组合：确定性 + 只读契约 + 路径锚定（Python 编排，避免 shell 链断掉）。"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from inference.console import use_utf8_console  # noqa: E402

use_utf8_console()

PY = sys.executable
OUT = ROOT / "out"


def run(*args):
    return subprocess.run([PY, "main.py", *args], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8")


def snap(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.iterdir()) if p.is_file()}


# ---- 1. 两次 run ----------------------------------------------------------
run("run")
a = snap(OUT)
run("run")
b = snap(OUT)
same = [n for n in sorted(a) if a[n] == b[n]]
diff = [n for n in sorted(a) if a[n] != b[n]]
print(f"[确定性] {len(a)} 个产物：{len(same)} 个逐位相同；不同：{diff}")

# ---- 2. 只读命令 + out/ 不变 ---------------------------------------------
before = snap(OUT)
codes = {}
for c in ("check", "show", "visualize", "verify"):
    codes[c] = run(c).returncode
for c in ("check", "verify", "show", "visualize"):
    codes[c + " --json"] = run(c, "--json").returncode
after = snap(OUT)
print(f"[只读] rc = {codes}")
moved = sorted(n for n in before if before[n] != after.get(n))
print(f"[只读] out/ 变化：{moved or '（无）'}")

# ---- 3. --json 的 stdout 只许有一段 JSON ----------------------------------
for c in ("check", "show", "verify", "visualize"):
    p = run(c, "--json")
    try:
        json.loads(p.stdout)
        print(f"[--json] {c}: 合法 JSON ✔")
    except Exception as exc:
        print(f"[--json] {c}: ❌ {exc} / stdout={p.stdout[:120]!r}")

# ---- 4. 从工作区根调用（路径锚定）----------------------------------------
ws = ROOT.parent
codes = {}
for c in ("run", "check", "show", "visualize", "verify", "dump-config"):
    codes[c] = subprocess.run(
        [PY, str(ROOT / "main.py"), c], cwd=ws,
        capture_output=True, text=True, encoding="utf-8").returncode
print(f"[锚定] 从工作区根调用 rc = {codes}")

# ---- 5. 锚定是恒等变换 ----------------------------------------------------
sys.path.insert(0, str(ROOT / "src"))
from inference import cli  # noqa: E402

parser = cli.build_parser()
for argv in (["run"], ["check"], ["show"], ["visualize"], ["verify"],
             ["dump-config"]):
    ns = parser.parse_args(argv)
    cli._fill_defaults(ns)
    keys = [k for k in ("config", "out") if hasattr(ns, k)]
    before_a = {k: str(getattr(ns, k)) for k in keys}
    cli._anchor_args(ns)
    after_a = {k: str(getattr(ns, k)) for k in keys}
    ok = all(pathlib.Path(before_a[k]).resolve() == pathlib.Path(after_a[k]).resolve()
             for k in keys)
    print(f"[锚定] {argv[0]:12s} 恒等={ok}  {after_a}")
