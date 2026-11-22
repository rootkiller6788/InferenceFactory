"""报告渲染：**只吃产物字典**，不吃运行期对象。

⚠️ `render(products)` 必须是**产物内容的纯函数**：
- 不读世界、不读模型、不读配置（`config.json` 在产物里，从它读）
- 不依赖 dict 的**插入顺序**（产物经 JSON 往返后键序是字典序）
  ⇒ 所有遍历一律 `sorted(...)`
"""

from __future__ import annotations

import html as _html
import json
from typing import Any

from .cost import human_bytes

_L: list[str] = []


def render(products: dict[str, Any]) -> tuple[str, str]:
    md = _render_md(products)
    return md, _render_html(md, products)


# --------------------------------------------------------------------- 小工具
def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out


def _num(v: Any, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return "∞" if v > 0 else "-∞"
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _dump(o: Any) -> str:
    return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(", ", ": "))


# --------------------------------------------------------------------- Markdown
def _render_md(p: dict[str, Any]) -> str:
    L: list[str] = []
    s = p["summary.json"]
    v = p["version.json"]
    cfg = p["config.json"]["resolved"]

    L.append("# InferenceFactory 运行报告")
    L.append("")
    L.append(f"- 版本 **`{v['version']}`**（协议指纹 `{v['protocol_digest']}`）")
    L.append(f"- 矩阵 {s['n_models']} 模型 × {s['n_hardware']} 机器 × "
             f"{s['n_configs']} 配置 = **{s['n_cells']} 格**（缺口 "
             f"**{s['matrix_gap']}**）")
    L.append(f"- 可行 **{s['n_feasible']}** 格；不可行原因 "
             f"**{s['n_infeasible_reasons']}** 种：{_dump(s['infeasible_reasons'])}")
    L.append(f"- 双实现互校 **{s['crosscheck']['n_agree']}/{s['crosscheck']['n_checked']}**"
             f"（flops 最大相对误差 {s['crosscheck']['flops']['max_rel_err']:.2e}；"
             f"排队 Erlang-C vs 模拟 "
             f"{s['crosscheck']['queue']['n_agree']}/{s['crosscheck']['queue']['n_checked']}，"
             f"ΔP(等待) ≤ {s['crosscheck']['queue']['max_abs_err']:.4f}）")
    L.append(f"- 门槛 **{s['acceptance']['n_passed']}/{s['acceptance']['n']}** 通过；"
             f"对抗用例 **{s['adversarial']['n_held']}/{s['adversarial']['n_cases']}** 成立；"
             f"期望 **{s['expectations']['n_held']}** 成立 / "
             f"**{s['expectations']['n_refuted']}** 被推翻")
    L.append(f"- 部署决策 **{s['n_decisions']}/{s['n_deployments']}** 个；"
             f"被 SLO 淘汰 **{s['n_slo_exclusions']}** 次；"
             f"赢过的 backend **{s['n_backend_wins']}** 个")
    L.append("")

    L.extend(_section_pipeline(p, cfg))
    L.append("")
    L.extend(_section_inspect(p))
    L.append("")
    L.extend(_section_feasible(p, cfg))
    L.append("")
    L.extend(_section_curves(p))
    L.append("")
    L.extend(_section_decisions(p, cfg))
    L.append("")
    L.extend(_section_expectations(p))
    L.append("")
    L.extend(_section_acceptance(p))
    L.append("")
    L.extend(_section_adversarial(p))
    L.append("")
    L.extend(_section_artifacts(p))
    return "\n".join(L) + "\n"


def _section_pipeline(p: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    L = ["## 1. 主流水线十步（每步一个产物）", ""]
    L.extend(_table(
        ["步", "名称", "产物", "这一步回答什么"],
        [["1", "Inspect", "`inspect.json`",
          "模型多大、KV 每 token 多贵、**权重放不放得下**"],
         ["2", "Select Backend", "`selection.json`",
          "哪些后端在这台机器上**根本不可能**（没设备 / 权重太大）"],
         ["3", "Convert / Load", "`convert.json`", "量化后体积、转换要写多少字节"],
         ["4", "Warmup", "`warmup.json`", "常驻基线（权重 + 框架）占掉多少预算"],
         ["5", "Benchmark", "`matrix.json`", "全矩阵 **1944 格**的 TTFT/TPOT/tokens·s⁻¹/VRAM"],
         ["6", "Stress Test", "`stress.json`", "能扛住多大到达率；扛不住是**排队**还是**容量**"],
         ["7", "Profile", "`scaling.json`", "三条 scaling 曲线（这个项目的核心输出）"],
         ["8", "Compare", "`compare.json`", "可行域 + Pareto + **在 SLO 内最大化吞吐**的决策"],
         ["9", "Package", "`artifact.json`", "带校验和的可部署声明"],
         ["10", "Deploy", "`deploy.json`", "按 Artifact 自己的读数复核 SLO + 回滚条件"]]))
    return L


def _section_inspect(p: dict[str, Any]) -> list[str]:
    L = ["## 2. 模型与机器（步 1 Inspect）", ""]
    rows = []
    for r in p["inspect.json"]["rows"]:
        fit = "、".join(sorted(q for q, ok in r["weights_fit"].items() if ok))
        nofit = "、".join(sorted(q for q, ok in r["weights_fit"].items() if not ok))
        rows.append([f"`{r['deployment']}`", _num(r["params_b"], 2) + " B",
                     f"{r['n_layers']}×{r['hidden']}", _num(r["gqa_ratio"], 1),
                     human_bytes(r["budget_bytes"]),
                     fit or "（无）", nofit or "（无）"])
    L.extend(_table(["部署单元", "参数量", "层×隐藏", "GQA", "设备预算",
                     "权重放得下", "放不下"], rows))
    return L


def _section_feasible(p: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    L = ["## 3. 可行域：一条双曲线 `B·C ≤ K`", ""]
    L.append("非分页后端按 `max_len` **预留**连续 KV ⇒ 占用 `∝ B·C`，"
             "于是可行域在 (batch, context) 平面上是一条双曲线。")
    L.append("分页后端（vLLM）只按**实际用掉**的 token 分配 ⇒ 与 `C` 无关。")
    L.append("")
    rows = []
    for d in p["compare.json"]["per_deployment"]:
        for key, h in sorted(d.get("hyperbolas", {}).items()):
            if not h["applies"]:
                rows.append([f"`{d['deployment']}`", key, "分页 ⇒ 与 C 无关",
                             _num(h["kv_per_token"] / 1024, 1) + " KB/token",
                             _num(h["max_batch_tokens"] / 1000, 1) + " K tokens"])
                continue
            rows.append([f"`{d['deployment']}`", key,
                         f"`B·C ≤ {h['max_bc_tokens']:,.0f}`",
                         _num(h["kv_per_token"] / 1024, 1) + " KB/token",
                         human_bytes(h["budget_bytes_at_batch1"])])
    L.extend(_table(["部署单元", "后端｜量化", "可行域", "KV 每 token", "预算/占用"],
                    rows[:12]))
    L.append("")
    L.append(f"（共 {len(rows)} 条；表里只列前 12 条，完整数据在 `compare.json`）")
    return L


def _section_curves(p: dict[str, Any]) -> list[str]:
    L = ["## 4. 三条 scaling 曲线（步 7 Profile）", ""]
    sc = p["scaling.json"]
    L.append(f"固定点：`{_dump(sc['pin'])}`；"
             f"{sc['n_series']} 条序列 × 3 轴 = **{sc['n_points']}** 个点。")
    L.append("")
    for axis in ("batch", "context", "arrival"):
        meta = sc["axes"][axis]
        L.append(f"### 4.{('batch', 'context', 'arrival').index(axis) + 1} "
                 f"{axis} 轴（{meta['x_label']}，其余固定 "
                 f"`{_dump(meta['pinned'])}`）")
        L.append("")
        rows = []
        for s in sorted(sc["series"], key=lambda x: x["series_id"])[:8]:
            pts = s["curves"][axis]
            cells = []
            for q in pts:
                if not q["feasible"]:
                    cells.append(f"{q['x']}:✗{q['reason'][:4]}")
                elif axis == "batch":
                    cells.append(f"{q['x']}:{q['tokens_per_s']:.0f}")
                elif axis == "context":
                    cells.append(f"{q['x']}:{q['tpot_ms']:.1f}ms")
                else:
                    cells.append(f"{q['x']}:{q['p99_ms']:.0f}ms")
            rows.append([f"`{s['series_id']}`", " ".join(cells)])
        L.extend(_table(["序列", "x → 读数（✗ = 不可行）"], rows))
        L.append("")
    return L


def _section_decisions(p: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    L = ["## 5. 部署决策（步 8 Compare）", ""]
    L.append("决策规则：**在 SLO 内最大化吞吐**（SLO 是硬门槛，不是第四个目标）。")
    L.append("")
    rows = []
    for d in p["compare.json"]["per_deployment"]:
        dec = d.get("decision")
        bt = d.get("best_throughput")
        if dec is None:
            rows.append([f"`{d['deployment']}`", "—", "**无决策**",
                         f"SLO 挡住={d.get('slo_blocked')}", "—", "—"])
            continue
        r = dec["readings"]
        thr_gap = ("（同点）" if bt and bt["config_id"] == dec["config_id"]
                   else f"（吞吐最优 {bt['readings']['tokens_per_s']:,.0f} 被 SLO 淘汰）"
                   if bt else "")
        rows.append([f"`{d['deployment']}`", dec["backend"], dec["quant"],
                     f"b={dec['batch']} ctx={dec['context']} λ={dec['arrival_rate']:g}",
                     f"{r['tokens_per_s']:,.0f} tok/s / P99 {r['p99_ms']:,.0f} ms",
                     thr_gap])
    L.extend(_table(["部署单元", "后端", "量化", "配置", "读数", "与吞吐最优的关系"], rows))
    return L


def _section_expectations(p: dict[str, Any]) -> list[str]:
    L = ["## 6. 设计期望（被推翻的是产出）", ""]
    e = p["expectations.json"]
    L.append(f"**{e['n_held']} 成立 / {e['n_refuted']} 被推翻 / "
             f"{e['n_unmeasurable']} 算不出来**")
    L.append("")
    for r in e["rows"]:
        mark = {"held": "成立", "refuted": "**推翻**", "unmeasurable": "算不出"}[r["verdict"]]
        L.append(f"- [{mark}] `{r['id']}` —— {r['claim']}")
        L.append(f"  - {r['detail']}")
        if r["observed"]:
            L.append(f"  - 观测：`{_dump(r['observed'])}`")
    return L


def _section_acceptance(p: dict[str, Any]) -> list[str]:
    L = ["## 7. 门槛（可调达标线）", ""]
    a = p["acceptance.json"]
    L.append(f"**{a['n_passed']}/{a['n']} 通过**")
    L.append("")
    rows = [[("OK" if r["passed"] else "**FAIL**"), f"`{r['key']}`", r["label"],
             f"{r['op']} {_num(r['limit'])}", _num(r["observed"])]
            for r in a["rows"]]
    L.extend(_table(["", "门槛", "含义", "要求", "实测"], rows))
    return L


def _section_adversarial(p: dict[str, Any]) -> list[str]:
    L = ["## 8. 对抗用例（不可调的不变量）", ""]
    a = p.get("adversarial.json")
    if not a:
        L.append("（对抗层尚未跑完 —— 这一步的产物在第二趟才齐）")
        return L
    L.append(f"**{a['n_held']}/{a['n_cases']} 成立**（配置里没有阈值 ⇒ "
             "它们只能被代码缺陷打破）")
    L.append("")
    rows = [[("成立" if r["held"] else "**违反**"), f"`{r['name']}`",
             r["detail"][:110]] for r in a["cases"]]
    L.extend(_table(["", "用例", "结论"], rows))
    return L


def _section_artifacts(p: dict[str, Any]) -> list[str]:
    L = ["## 9. 部署 Artifact（步 9–10）", ""]
    rows = []
    for a in p["artifact.json"]["artifacts"]:
        if a["status"] != "ok":
            rows.append([f"`{a['deployment']}`", "—", a["reason"], "—"])
            continue
        body = a["artifact"]
        rows.append([f"`{a['deployment']}`", f"`{a['checksum']}`",
                     f"{body['backend']} / {body['quant']}",
                     f"max_batch={body['serving']['max_batch']}, "
                     f"max_ctx={body['serving']['max_context']}, "
                     f"λ={body['serving']['target_arrival_rate']:g}"])
    L.extend(_table(["部署单元", "校验和", "后端 / 量化", "serving 配置"], rows))
    L.append("")
    rows = []
    for r in p["deploy.json"]["rows"]:
        rows.append([f"`{r['deployment']}`",
                     "是" if r.get("slo_met") else "否",
                     _num(r.get("p99_ms"), 1), _num(r.get("tpot_ms"), 2),
                     _num(r.get("tokens_per_s"), 0),
                     r.get("rollback_if", "—")])
    L.extend(_table(["部署单元", "SLO 达标", "P99 (ms)", "TPOT (ms)",
                     "tokens/s", "回滚条件"], rows))
    return L


# --------------------------------------------------------------------- HTML
def _render_html(md: str, products: dict[str, Any]) -> str:
    """极简 md → html：**表格**与**标题**是唯一需要结构的部分。

    ⚠️ 不引外部 CSS/JS（离线可读），不做语法高亮。
    """
    out: list[str] = []
    in_ul = False
    in_table = False
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            if not in_table:
                out.append("<table>")
                in_table = True
                # 表头
                cells = [c.strip() for c in ln.strip("|").split("|")]
                out.append("<thead><tr>" + "".join(
                    f"<th>{_html.escape(c)}</th>" for c in cells) + "</tr></thead>")
                i += 2          # 跳过分隔行
                out.append("<tbody>")
                continue
            if ln.strip().startswith("|---") or set(ln.replace("|", "").strip()) <= {"-"}:
                i += 1
                continue
            cells = [c.strip() for c in ln.strip("|").split("|")]
            out.append("<tr>" + "".join(f"<td>{_html.escape(c)}</td>"
                                        for c in cells) + "</tr>")
            i += 1
            continue
        if in_table:
            out.append("</tbody></table>")
            in_table = False
        if ln.startswith("#"):
            lvl = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{lvl}>{_html.escape(ln.lstrip('# '))}</h{lvl}>")
        elif ln.startswith("- "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{_html.escape(ln[2:])}</li>")
        else:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if ln.strip():
                out.append(f"<p>{_html.escape(ln)}</p>")
        i += 1
    if in_table:
        out.append("</tbody></table>")
    if in_ul:
        out.append("</ul>")
    title = f"InferenceFactory {products['version.json']['version']}"
    return ("<!DOCTYPE html>\n<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            f"<title>{_html.escape(title)}</title>"
            "<style>body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;"
            "max-width:1180px;margin:2rem auto;padding:0 1rem;line-height:1.6}"
            "table{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.86rem}"
            "th,td{border:1px solid #d0d7de;padding:.3rem .5rem;text-align:left}"
            "th{background:#f6f8fa}h1,h2,h3{margin-top:1.6rem}"
            "code{background:#f0f2f4;padding:.1rem .3rem;border-radius:3px}</style>"
            f"</head><body>{''.join(out)}</body></html>\n")
