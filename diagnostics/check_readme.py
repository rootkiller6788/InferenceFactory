"""README 数字核对：附录 B 里的每一个数，都必须能**从最终 `out/` 重新导出**。

为什么要写这个
--------------
「每个数字都从最终 `out/` 重新导出」是一条**自觉**，而自觉会失效：
改一次配置、重跑一次流水线，附录 B 就有一半的数字变成**上一次运行的读数**，
而它们看起来完全正常。这个脚本是那条自觉的机器化。

做法
----
1. 把 `out/*.json` 里的**全部数字**收进一个池子
   —— 包括嵌在文案里的数字（`detail` / `note` 里的「0.0084 倍」也算读数）；
2. 从 README 的**附录 B** 里抠出全部数字 token；
3. 每个 token 必须能被池子里某个数**按它自己的精度舍入**得到，
   或能被一个**派生量**（由池子里的数算出来）解释；
   否则必须落在一张**显式声明**的表里，且每项写理由；
4. 再核几条**复合断言**（版本、指纹、门槛计数、决策坐标…）确实出现在 README 里；
5. 核 A.3 的「写盘」列与每个子命令的 `--help` **逐条一致**；
6. 扫附录 C 的编号：最大编号、重复、连续性、以及**与上一项目接得上**。

判据是「**能被舍入得到**」而不是「字符串相等」—— 因为附录 B 会做
`%.1f` / `%.4f` 这类排版舍入，而产物里是全精度。

⚠️ 已知的弱点（要如实说，不能假装它很强）：
池子里到处都有的**小整数**（0 / 1 / 2 / 3 / 8 …）几乎总能对上，
所以「写错一个小整数」它抓不住。它抓的是**有信息量的数**
（1944、148、0.0495、指纹、决策坐标…）。⇒ 所以下面还配了几条**复合断言**，
专盯那几个「差一点点就骗过去」的数。

用法（在项目根）：
    python diagnostics/check_readme.py
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inference.console import use_utf8_console  # noqa: E402

use_utf8_console()

README = ROOT / "README.md"

#: 数字 token：前后不许紧贴字母或小数点
#: （于是 `b8|c2048` 里的 8/2048 算，`p99_ms`、`rtx4060-8g`、`v1-2ebc7658daa7` 不算）。
TOKEN = re.compile(r"(?<![\w.])[-−–]?\d+(?:\.\d+)?(?:[eE][-−]?\d+)?(?![\w.])")

#: 手测 / 历史读数 —— **每一项都必须写理由**，否则这张表会变成
#: 「随手加数字的地方」，而它迟早会失控。
#:
#: 本项目**刻意留空**：附录 B 里的每一个数字都能从最终 `out/` 重新导出。
#: 留空不是省事，是更强的承诺 —— 一旦这里出现第一项，
#: 「附录 B 全部可导出」就降级成「除了表里那几项」。
MANUAL: dict[str, str] = {}

#: 附录 C 的编号必须**紧接上一项目**。上一项目 `L7-kvcachefactory` 止于 388。
C_FIRST = 389


def walk_numbers(node, sink: list) -> None:
    """把产物里的数字收进池子 —— **包括嵌在文案里的数字**。

    `detail` / `note` / `claim` 这些字符串也是产物的一部分，里面的数字
    （「0.0084 倍」、「36 段持平」）同样是产物里的读数。
    只收数字叶子会漏掉它们，然后逼着人去手抄一张例外表 —— 那张表迟早会失控。

    ⚠️ **字典的键不收**（只走 `values()`）。键是**标签**，不是读数。
    """
    if isinstance(node, dict):
        for v in node.values():
            walk_numbers(v, sink)
    elif isinstance(node, list):
        for v in node:
            walk_numbers(v, sink)
    elif isinstance(node, bool):
        return
    elif isinstance(node, (int, float)):
        sink.append(float(node))
    elif isinstance(node, str):
        for tok in TOKEN.findall(node):
            try:
                sink.append(float(tok.replace("−", "-").replace("–", "-")))
            except ValueError:
                pass


def sig_interval(tok: str) -> tuple[float, float]:
    """token 自己的「半个末位」区间 —— 由它写了多少位有效数字推出来。"""
    t = tok.replace("−", "-").replace("–", "-")
    val = float(t)
    body = t.lower()
    mant = body.split("e")[0].lstrip("-")
    int_part, frac = mant.split(".") if "." in mant else (mant, "")
    digits = (int_part + frac).lstrip("0")
    s = len(digits) or 1
    if val == 0.0:
        return (-5e-13, 5e-13)
    exp = int(body.split("e")[1]) if "e" in body else 0
    e10 = math.floor(math.log10(abs(val))) + exp
    half = 0.5 * 10 ** (e10 - s + 1)
    return (val - half, val + half)


#: 两个**标记词**：它们是「分类标签」，不是「写目标」。
_MARKERS = ("会写盘", "只读")

#: 写目标白名单 —— 本项目只有这两个写根（产物 `out/`、口径 `configs/`）。
_WRITE_ROOTS = ("out/", "configs/")


def _write_target(cell: str) -> str:
    """从说明里抠出**写目标**；没有就返回 `""`。

    ⚠️⚠️ 必须先剔掉标记词再找 —— 标记词 `会写盘` **自己**就含「写盘」，
    所以「说明里有没有写盘字样」这条判据在 `"会写盘" in desc` 为真时**恒真**。
    它因此从写下那天起就没生效过（实测 2026-10-07：工作区里 8 个项目
    的 `cli_contract` 都带这一行；其中 3 家已改成「先剔标记」但仍是**动词**判据，
    5 家还是原样）。一修好就立刻抓出真缺陷。
    """
    bare = cell
    for mk in _MARKERS:
        bare = bare.replace(mk, "")
    for root in _WRITE_ROOTS:
        if root in bare:
            return root
    return ""


def _help_desc(cmd: str) -> tuple[int, str]:
    """跑 `main.py <cmd> --help`，把 description 段拼成一行。

    返回 `(returncode, 说明)`；rc != 0 时说明为空串。
    """
    proc = subprocess.run([sys.executable, "main.py", cmd, "--help"],
                          cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        return proc.returncode, ""
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines()]
    i = next((k for k, ln in enumerate(lines) if ln.startswith("usage:")), 0)
    while i < len(lines) and lines[i]:
        i += 1
    start = i + 1
    end = next((k for k, ln in enumerate(lines)
                if k >= start and ln.startswith("options:")), len(lines))
    return 0, " ".join(ln for ln in lines[start:end] if ln)


def cli_contract(text: str,
                 help_of: Callable[[str], tuple[int, str]] | None = None
                 ) -> list[str]:
    """把 README 的 A.3 表**对着 `--help` 机器核**。

    ⚠️ 为什么不能靠「跑一遍看文件变没变」来核：`dump-config` 写回的是
    **同一份内容** ⇒ 任何基于哈希的核对都会把它判成只读。
    **哈希能证明「没变」，不能证明「没写」。**

    ⚠️ 而且「`--help` 里到底有没有那个标记」这件事，**外部脚本也核不出来**：
    它是拿 `--help` 去核 README 的，两边一起缺标记就等于两边一起漂移。
    ⇒ 那一条由流水线里的对抗用例 `subcommand_help_declares_readwrite` 钉住，
    这里只负责核「README 与 `--help` 说的是不是同一件事」。
    """
    rows: dict[str, str] = {}
    in_table = False
    for line in text.splitlines():
        if line.startswith("## A.3 CLI 契约"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if in_table:
            m = re.match(r"\|\s*`([a-z-]+)`\s*\|(.*?)\|", line)
            if m:
                rows[m.group(1)] = m.group(2)
    if not rows:
        return ["A.3 表没解析出来 —— 契约没被核到（这本身就是失败）"]
    if help_of is None:
        help_of = _help_desc

    problems: list[str] = []
    for cmd in sorted(rows):
        cell = rows[cmd]
        rc, desc = help_of(cmd)
        if rc != 0:
            problems.append(f"`{cmd} --help` rc={rc}")
            continue

        doc_write = "会写盘" in desc
        doc_readonly = "只读" in desc
        readme_write = "会写盘" in cell
        readme_readonly = "只读" in cell
        if doc_write == doc_readonly:
            problems.append(f"`{cmd}`：`--help` 里 `会写盘`/`只读` 两个标记"
                            f"必须**恰好出现一个** → {desc!r}")
        if readme_write == readme_readonly:
            problems.append(f"`{cmd}`：README 的「写盘」列里 `会写盘`/`只读` "
                            f"必须**恰好出现一个** → {cell.strip()!r}")
        if doc_write != readme_write:
            problems.append(
                f"`{cmd}`：README 说「{'会写盘' if readme_write else '只读'}」，"
                f"`--help` 说「{'会写盘' if doc_write else '只读'}」 → {desc!r}")
        if doc_write and not _write_target(desc):
            problems.append(f"`{cmd}`：标了「会写盘」但没说写到哪 → {desc!r}")
    return problems


def appendix_c(text: str) -> list[str]:
    """附录 C 的编号：最大、重复、连续、与上一项目接得上。

    ⚠️ 必须**只扫附录 C 那一段**：拿全文扫 `^### (\\d+)\\.` 会把
    `### 9.1`、`### 10.1` 这些**节号**算进来，最小编号印成 9/10 ——
    一个看起来像「编号连续」的假读数。
    """
    problems: list[str] = []
    sec = text[text.index("# 附录 C｜踩过的坑"):]
    nums = [int(n) for n in re.findall(r"^### (\d+)\.", sec, re.M)]
    if not nums:
        return ["附录 C 里一个 `### NNN.` 标题都没有"]
    dup = sorted({n for n in nums if nums.count(n) > 1})
    if dup:
        problems.append(f"附录 C 有重复编号：{dup}")
    if min(nums) != C_FIRST:
        problems.append(f"附录 C 的第一条是 {min(nums)}，应该是 {C_FIRST}"
                        f"（上一项目 L7-kvcachefactory 止于 {C_FIRST - 1}）")
    gaps = [b for a, b in zip(nums, nums[1:]) if b != a + 1]
    if gaps:
        problems.append(f"附录 C 编号不连续，断在：{gaps}")
    # 分组标题与区间形态也要看 —— 只按 `^### N.` 扫会漏整段
    sec_groups = re.findall(r"^#{3,4} *[A-Z]\.\d+ ", sec, re.M)
    if sec_groups:
        problems.append(f"附录 C 里混进了分组标题：{sec_groups}")
    return problems


# ---------------------------------------------------------------------------
# 自证：把每条判据弄坏一次，看它翻不翻脸
#: 形状与真表一致的**最小 A.3 表**（本项目的解析取**第 2 格** = 写列）。
_FAKE_A3 = (
    "## A.3 CLI 契约\n\n"
    "| 子命令 | 写盘 | 说明 |\n"
    "| --- | --- | --- |\n"
    "| `run` | **会写盘** `out/` | 端到端 |\n"
    "| `check` | **只读** | 核对 |\n"
    "\n"
    "## A.4 别的\n"
)

#: 形状与真附录 C 一致的**最小附录 C**（首号 = `C_FIRST`）。
_FAKE_C = ("# 附录 C｜踩过的坑\n\n"
           f"### {C_FIRST}. 甲\n"
           f"### {C_FIRST + 1}. 乙\n"
           f"### {C_FIRST + 2}. 丙\n")
_FAKE_C_DUP = _FAKE_C.replace(f"### {C_FIRST + 2}.", f"### {C_FIRST + 1}.")
_FAKE_C_SHIFT = _FAKE_C.replace(f"### {C_FIRST}.", f"### {C_FIRST - 1}.")
_FAKE_C_GAP = _FAKE_C.replace(f"### {C_FIRST + 2}.", f"### {C_FIRST + 3}.")


def _fake_help(cmd: str) -> tuple[int, str]:
    """假的 `--help` 取用器：run 会写盘（且**说了**写目标），其余只读。"""
    if cmd == "run":
        return 0, "执行全流水线并落盘（会写盘：写 out/）"
    return 0, "核对 out/ 与版本自洽（只读）"


def check_selftest(text: str) -> tuple[int, int, list[str], list[str]]:
    """**每条判据都弄坏一次输入，看它翻不翻脸。**

    ⚠️ 从不翻脸的核对脚本 = 没有核对脚本。
    ⚠️ 只有**显式判红**（判据自己返回了非空 problems）才算证明：
    **抛异常**不算 —— 那说明判据连跑都跑不起来，与「它发现了问题」不是一回事。
    装载期崩溃同理（脚本都 import 不进来，自然什么也核不了）。
    ⚠️ 假表 `_FAKE_A3` 的**形状必须与真表的解析同体**：本项目取第 2 格，
    所以写列就放在第 2 格。放错格会让「期望红」的探针**因为 cell 里
    没有标记而恒红** —— 假红，与真红在输出里长得一模一样。

    返回 `(注册条数, 证明条数, 抛异常清单, 未翻脸清单)`。
    """
    proved: list[str] = []
    raised: list[str] = []
    failed: list[str] = []

    def expect_red(fn):
        """判据必须**返回非空**。异常**穿出去**，由下面归类成「抛异常」。"""
        return bool(fn())

    def expect_green(fn):
        return not fn()

    probes: list[tuple[str, Callable[[], bool]]] = [
        ("写目标：只写标记词不算写了目标",
         lambda: _write_target("（会写盘）") == ""),
        ("写目标：写了 `out/` 才算写了目标",
         lambda: _write_target("（会写盘：写 out/）") == "out/"),
        ("sig_interval 按有效位给半末位区间",
         lambda: abs(sig_interval("1.0")[0] - 0.95) < 1e-12
                 and abs(sig_interval("1.0")[1] - 1.05) < 1e-12),
        ("A.3：找不到那一节要报",
         lambda: expect_red(lambda: cli_contract("# 没有 A.3\n"))),
        ("A.3：README 说反了要报",
         lambda: expect_red(lambda: cli_contract(
             _FAKE_A3, help_of=lambda c: (0, "核对 out/（只读）")))),
        ("A.3：两个标记同时出现要报",
         lambda: expect_red(lambda: cli_contract(
             _FAKE_A3, help_of=lambda c: (0, "会写盘 只读")))),
        ("A.3：标了会写盘却不说写到哪要报",
         lambda: expect_red(lambda: cli_contract(
             _FAKE_A3, help_of=lambda c: (0, "端到端跑一遍并写盘（会写盘）")))),
        ("A.3：说了写目标就不报",
         lambda: expect_green(lambda: cli_contract(_FAKE_A3, help_of=_fake_help))),
        ("A.3：`--help` 跑不起来要报",
         lambda: expect_red(lambda: cli_contract(
             _FAKE_A3, help_of=lambda c: (2, "")))),
        ("附录 C：正常编号不报",
         lambda: expect_green(lambda: appendix_c(_FAKE_C))),
        ("附录 C：重复编号要报",
         lambda: expect_red(lambda: appendix_c(_FAKE_C_DUP))),
        ("附录 C：首号不对要报",
         lambda: expect_red(lambda: appendix_c(_FAKE_C_SHIFT))),
        ("附录 C：断号要报",
         lambda: expect_red(lambda: appendix_c(_FAKE_C_GAP))),
    ]

    for name, fn in probes:
        try:
            ok = bool(fn())
        except Exception as exc:                      # noqa: BLE001
            raised.append(f"{name}（{type(exc).__name__}: {exc}）")
            continue
        (proved if ok else failed).append(name)

    return len(probes), len(proved), raised, failed


def main() -> int:
    products = {f.name: json.loads(f.read_text(encoding="utf-8"))
                for f in sorted((ROOT / "out").glob("*.json"))}
    if not products:
        print("❌ out/ 里没有产物 —— 先跑 `python main.py run`")
        return 1

    pool: list[float] = []
    for doc in products.values():
        walk_numbers(doc, pool)
    pool = sorted(set(pool))

    # ---- 派生量（由池子里的数**算出来**，不是手抄的）----------------------
    derived: dict[str, str] = {}

    def derive(label: str, value: float) -> None:
        derived[f"{value:.10g}"] = label

    sm = products["summary.json"]
    # B.2 的 born 种类数 / B.0 的不可行原因合计
    derive("不可行合计", float(sum(sm["infeasible_reasons"].values())))
    derive("可行+不可行", float(sm["n_feasible"] + sum(sm["infeasible_reasons"].values())))
    # B.7 的产物个数（**全部文件**，含 .md / .html，不是 len(products)）
    derive("out/ 里的文件个数",
           float(len([p for p in (ROOT / "out").rglob("*") if p.is_file()])))
    derive("out/ 里的 json 个数", float(len(products)))
    # B.1 的轴长乘积（3×2×4×3×3×3×3 去重后 324）
    derive("轴长乘积", float(len(products["models.json"]["models"])
                            * len(products["hardware.json"]["hardware"])
                            * len(products["backends.json"]["backends"])
                            * 3 * 3 * 3 * 3))
    # scaling 每序列的点数（B.5 说「3 条曲线 × 60 序列 = 1620 点」）
    sc = products["scaling.json"]
    derive("scaling 每序列点数",
           float(sc["n_points"] / sc["n_series"]) if sc["n_series"] else 0.0)
    # 变异测试类型分布（B.6）
    mut = json.loads((ROOT / "diagnostics" / "mutation.json").read_text(encoding="utf-8"))
    derive("变异-通用", float(mut.get("n_generic", 0)))
    derive("变异-定向", float(mut.get("n_targeted", 0)))
    derive("变异总数校验", float(mut.get("n_cases", 0) - mut.get("n_proven", 0)))

    derived_vals = [float(k) for k in derived]

    # ---- 抠 README 附录 B 的数字 ----------------------------------------
    text = README.read_text(encoding="utf-8")
    start = text.index("# 附录 B｜实测结果")
    end = text.index("# 附录 C｜踩过的坑")
    tokens = sorted(set(TOKEN.findall(text[start:end])))

    def matched(tok: str) -> bool:
        lo, hi = sig_interval(tok)
        return (any(lo <= v <= hi for v in pool)
                or any(lo <= v <= hi for v in derived_vals))

    unmatched = [t for t in tokens if not matched(t)]
    manual_hit = [t for t in unmatched if t in MANUAL]
    bad = [t for t in unmatched if t not in MANUAL]

    # ---- 复合断言：这几条必须出现，且必须与产物一致 ----------------------
    acc = products["acceptance.json"]
    adv = products["adversarial.json"]
    exp = products["expectations.json"]
    cmp_ = products["compare.json"]
    ver = products["version.json"]
    cc = products["crosscheck.json"]
    claims = [
        ("版本", ver["version"]),
        ("协议指纹", ver["protocol_digest"]),
        ("版本摘要文件数", f"{ver['n_versioned']}"),
        ("格数", f"{sm['n_cells']}"),
        ("可行格", f"{sm['n_feasible']}"),
        ("不可行原因种类", f"{sm['n_infeasible_reasons']}"),
        ("配置数", f"{sm['n_configs']}"),
        ("互校总数", f"{sm['crosscheck']['n_agree']}/{sm['crosscheck']['n_checked']}"),
        ("互校排队数", f"{sm['crosscheck']['queue']['n_agree']}/"
                      f"{sm['crosscheck']['queue']['n_checked']}"),
        ("门槛计数", f"{acc['n_passed']} / {acc['n']}"),
        ("对抗计数", f"{adv['n_held']} / {adv['n_cases']}"),
        ("期望计数", f"{exp['n_held']} 成立 / {exp['n_refuted']} 被推翻"),
        ("决策计数", f"{sm['n_decisions']} / {sm['n_deployments']}"),
        ("SLO 淘汰", f"{sm['n_slo_exclusions']}"),
        ("后端胜出数", f"{sm['n_backend_wins']}"),
        ("Pareto 总点数", f"{sm['n_pareto_total']}"),
        ("变异测试条数", f"{mut['n_cases']} / {mut['n_proven']}"),
        ("不可行原因合计", f"{sum(sm['infeasible_reasons'].values())}"),
        ("部署单元数", f"{sm['n_deployments']}"),
        ("scaling 序列数", f"{sc['n_series']}"),
        ("scaling 点数", f"{sc['n_points']}"),
    ]
    missing_claims = [(lab, val) for lab, val in claims if val not in text]

    print(f"产物 {len(products)} 个；数字池 {len(pool)} 个数；派生 {len(derived)} 个数")
    print(f"附录 B 里的数字 token：{len(tokens)} 个")
    print(f"  能从产物/派生量解释：{len(tokens) - len(unmatched)}")
    print(f"  落在手测/历史表里：{len(manual_hit)}  → {manual_hit}")
    print(f"  **解释不了**：{len(bad)}  → {bad}")
    print()
    if missing_claims:
        print("❌ 复合断言缺失：")
        for lab, val in missing_claims:
            print(f"   - {lab}：README 里没有 {val!r}")
    else:
        print(f"✅ 复合断言全部命中（{len(claims)} 条）")

    cli_bad = cli_contract(text)
    if cli_bad:
        print("\n❌ A.3 的「写盘」列与 `--help` 不一致：")
        for x in cli_bad:
            print(f"   - {x}")
    else:
        print("\n✅ A.3 的「写盘」列与每个子命令的 `--help` 逐条一致")

    c_bad = appendix_c(text)
    if c_bad:
        print("\n❌ 附录 C 编号有问题：")
        for x in c_bad:
            print(f"   - {x}")
    else:
        sec = text[text.index("# 附录 C｜踩过的坑"):]
        nums = [int(n) for n in re.findall(r"^### (\d+)\.", sec, re.M)]
        print(f"✅ 附录 C 编号连续、无重复：{min(nums)}–{max(nums)}"
              f"（上一项目止于 {C_FIRST - 1}）")

    # ---- 自证：每条判据都得证明自己**会翻脸** ---------------------------
    n_reg, n_proved, st_raise, st_fail = check_selftest(text)
    print(f"\n自证能红：注册 {n_reg} 条判据，证明 {n_proved} 条能翻脸")
    if st_raise:
        print(f"  ⚠️ **抛异常**（不算证明）：{st_raise}")
    if st_fail:
        print(f"  ❌ **未翻脸**（判据恒真）：{st_fail}")

    ok = (not bad and not missing_claims and not cli_bad and not c_bad
          and not st_raise and not st_fail)
    print("\n结论：" + ("**全部通过**" if ok else "**有核对未通过**"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
