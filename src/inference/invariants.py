"""正确性层：**门槛**（可调达标线）+ **对抗用例**（不可调的不变量）。

两层不能混：
- **门槛**：写在 `config.acceptance` 里，改了不用改代码。**可以被推翻**，
  推翻了是产出（「我以为 batch 能带来 10×，实测只有 6×」）。
- **对抗用例**：配置里**没有阈值** —— 它们是「无论门槛怎么定都必须成立」的
  结构性断言。它们只能被**代码缺陷**打破，不能被「调低要求」绕过。

⚠️ **恒真的判据比没有判据更危险** —— 它占着一个「这里验过了」的位置，
在报告里长得和别的判据一样。所以每一条都要被 `.scratch/prove_cases_can_fail.py`
**逐条证明能红**。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from . import emit as E
from .specs import HardwareSpec, ModelSpec

# --------------------------------------------------------------------- 工具
_TIME_SUFFIXES = ("_s", "_ms", "_us", "_seconds")

#: **口径层**产物：里面出现的时间键是**参数**（门槛 / 开销常数），不是测量。
_CATALOG_FILES: frozenset[str] = frozenset({"config.json", "backends.json"})

#: **模型预测**的性能类键（时间 + 速率），不是实测。
#: ⚠️ 白名单必须同时收**时间**和**速率**：`_s` 这个后缀既匹配 `service_s`
#: （时间）也匹配 `tokens_per_s`（速率，量纲是 1/秒）。只收前者的话，
#: 这条用例会去咬一个**吞吐读数** —— 那不是失真，那是这个项目的目标轴。
#: 白名单里出现任何 `elapsed` / `wall` / `measured` 才是真的失真。
_MODEL_TIME_KEYS: frozenset[str] = frozenset({
    # ---- 时间 ----
    "ttft_service_ms", "ttft_ms", "tpot_ms", "p99_ms",
    "prefill_s", "decode_s", "service_s", "service_with_wait_s",
    "wait_batch_s", "wq_mean_s", "wq_p99_s",
    "prefill_full_s", "decode_full_s", "service_full_s",
    "compute_s", "memory_s", "overhead_s", "transfer_s", "seconds",
    "convert_seconds", "warmup_seconds",
    # ---- 速率（**不是**时间，但同样带 `_s` 后缀）----
    "tokens_per_s", "delivered_tokens_per_s",
    "capacity_req_per_s", "served_req_per_s",
    # ---- 期望表里的**派生读数**（同样是模型预测出来的延迟差）----
    "max_gain_ms",
})


def _cells(p: dict[str, Any]) -> list[dict[str, Any]]:
    return p["matrix.json"]["cells"]


def _feas(p: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in _cells(p) if c["feasible"]]


def _series(p: dict[str, Any]) -> list[dict[str, Any]]:
    return p["scaling.json"]["series"]


def _groups(cells: Iterable[dict[str, Any]],
            keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for c in cells:
        out.setdefault(tuple(c[k] for k in keys), []).append(c)
    return out


def _rel(a: float, b: float) -> float:
    if b == 0:
        return 0.0 if a == 0 else float("inf")
    return abs(a - b) / abs(b)


# ===================================================================== 门槛
@staticmethod
def _obs_matrix_gap(p: dict[str, Any], c: dict[str, Any]) -> float:
    return float(p["matrix.json"]["matrix_gap"])


def _obs_n_cells(p: dict[str, Any], c: dict[str, Any]) -> float:
    return float(p["matrix.json"]["n_cells"])


def _obs_n_feasible(p: dict[str, Any], c: dict[str, Any]) -> float:
    return float(p["matrix.json"]["n_feasible"])


def _obs_n_reasons(p: dict[str, Any], c: dict[str, Any]) -> float:
    return float(len(p["matrix.json"]["infeasible_reasons"]))


def _obs_batch_speedup(p: dict[str, Any], c: dict[str, Any]) -> float:
    """batch 轴上吞吐的最大加速比（`tokens_per_s` 的可行端点之比）。"""
    best = 0.0
    for s in _series(p):
        pts = [q for q in s["curves"]["batch"] if q["feasible"]
               and q["tokens_per_s"]]
        if len(pts) < 2:
            continue
        lo = pts[0]["tokens_per_s"]
        hi = max(q["tokens_per_s"] for q in pts)
        if lo and lo > 0:
            best = max(best, hi / lo)
    return best


def _obs_context_tpot_growth(p: dict[str, Any], c: dict[str, Any]) -> float:
    best = 0.0
    for s in _series(p):
        pts = [q for q in s["curves"]["context"] if q["feasible"] and q["tpot_ms"]]
        if len(pts) < 2:
            continue
        lo = pts[0]["tpot_ms"]
        hi = max(q["tpot_ms"] for q in pts)
        if lo and lo > 0:
            best = max(best, hi / lo)
    return best


def _obs_p99_growth(p: dict[str, Any], c: dict[str, Any]) -> float:
    best = 0.0
    for s in _series(p):
        pts = [q for q in s["curves"]["arrival"] if q["feasible"] and q["p99_ms"]]
        if len(pts) < 2:
            continue
        lo = pts[0]["p99_ms"]
        hi = max(q["p99_ms"] for q in pts)
        if lo and lo > 0:
            best = max(best, hi / lo)
    return best


def _obs_crosscheck_flops(p: dict[str, Any], c: dict[str, Any]) -> float:
    return float(p["crosscheck.json"]["flops"]["max_rel_err"])


def _obs_crosscheck_bytes(p: dict[str, Any], c: dict[str, Any]) -> float:
    return float(p["crosscheck.json"]["bytes"]["max_rel_err"])


def _obs_queue_abs(p: dict[str, Any], c: dict[str, Any]) -> float:
    """排队互校：`P(等待>0)` 的解析值 vs 模拟频率，最大**绝对**误差。"""
    return float(p["crosscheck.json"]["queue"]["max_abs_err"])


def _obs_kv_share(p: dict[str, Any], c: dict[str, Any]) -> float:
    """可行域内，KV 占峰值占用的**最大**比例。

    ⚠️ 取「全可行域最大」而不是「最长 context 下」：后者问的是一个**更窄**
    的问题，而它恰好不成立 —— 见期望 `kv_dominates_at_long_context`。
    「KV 到底能不能成为显存主角」与「拉长 context 能不能让它成为主角」
    是**两个**问题，门槛只钉第一个（能），第二个交给期望去推翻。
    """
    best = 0.0
    for cell in _feas(p):
        m = cell["memory"]
        if m["peak_bytes"] > 0:
            best = max(best, m["kv_bytes"] / m["peak_bytes"])
    return best


def _obs_bound_variety(p: dict[str, Any], c: dict[str, Any]) -> float:
    kinds = {cell["readings"]["bound_decode"] for cell in _feas(p)}
    kinds |= {cell["readings"]["bound_prefill"] for cell in _feas(p)}
    return float(len(kinds))


def _obs_backend_wins(p: dict[str, Any], c: dict[str, Any]) -> float:
    return float(p["compare.json"]["n_backend_wins"])


def _obs_models_with_oom(p: dict[str, Any], c: dict[str, Any]) -> float:
    """**fp16 权重本身就放不下**的部署单元数（⇒ 必须量化或换机器）。

    ⚠️ 判据是「fp16 放不放得下」，不是「这个后端可不可行」：后者会因为
    「int4 放得下」而**永远为真** ⇒ 门槛恒过，而恒过的门槛与没有门槛
    无法区分。这条真正要守的是「显存约束在这批数据上到底 binding 不 binding」。
    """
    seen: set[str] = set()
    for r in p["inspect.json"]["rows"]:
        if not r["weights_fit"].get("fp16", True):
            seen.add(r["deployment"])
    return float(len(seen))


def _obs_slo_exclusions(p: dict[str, Any], c: dict[str, Any]) -> float:
    return float(p["compare.json"]["n_slo_exclusions"])


def _obs_pareto_points(p: dict[str, Any], c: dict[str, Any]) -> float:
    return float(sum(x.get("n_pareto", 0) for x in p["compare.json"]["per_deployment"]))


_THRESHOLDS: tuple[tuple[str, str, Callable[[dict, dict], float], str, str], ...] = (
    ("matrix_gap", "全枚举 ⇒ 缺口必须为 0", _obs_matrix_gap, "matrix_gap_max", "<="),
    ("n_cells", "矩阵规模", _obs_n_cells, "min_n_cells", ">="),
    ("n_feasible", "可行格数", _obs_n_feasible, "min_feasible_cells", ">="),
    ("n_infeasible_reasons", "不可行原因的种类数（可行性要是真维度）",
     _obs_n_reasons, "min_infeasible_reasons", ">="),
    ("batch_speedup", "batch 轴上吞吐的加速比", _obs_batch_speedup,
     "min_batch_speedup", ">="),
    ("context_tpot_growth", "context 拉长后 TPOT 的涨幅", _obs_context_tpot_growth,
     "min_context_tpot_growth", ">="),
    ("p99_growth_with_load", "到达率上来后 P99 的涨幅", _obs_p99_growth,
     "min_p99_growth_with_load", ">="),
    ("crosscheck_flops_rel", "双实现互校：flops", _obs_crosscheck_flops,
     "max_crosscheck_rel_err", "<="),
    ("crosscheck_bytes_rel", "双实现互校：bytes", _obs_crosscheck_bytes,
     "max_crosscheck_rel_err", "<="),
    ("queue_sim_abs", "排队：Erlang-C vs 离散事件模拟（`P(等待>0)` 绝对误差）",
     _obs_queue_abs, "max_queue_sim_abs_err", "<="),
    ("kv_share_peak", "KV 占峰值占用的最大比例（可行域内）", _obs_kv_share,
     "min_kv_share_peak", ">="),
    ("roofline_bound_variety", "矩阵里出现的 roofline bound 种类数",
     _obs_bound_variety, "min_roofline_bound_variety", ">="),
    ("n_backend_wins", "赢过的后端数（否则「选后端」是假的）",
     _obs_backend_wins, "min_backend_wins", ">="),
    ("n_models_with_oom", "权重就放不下的部署单元数", _obs_models_with_oom,
     "min_models_with_oom", ">="),
    ("n_slo_exclusions", "被 SLO 淘汰的「吞吐最优」次数", _obs_slo_exclusions,
     "min_slo_exclusions", ">="),
    ("n_pareto_points", "前沿点数合计", _obs_pareto_points,
     "min_pareto_points", ">="),
)


def run_acceptance(products: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    lim = cfg["acceptance"]
    rows = []
    for key, label, fn, limit_key, op in _THRESHOLDS:
        limit = float(lim[limit_key])
        try:
            obs = float(fn(products, cfg))
        except Exception as exc:                       # noqa: BLE001
            obs = float("nan")
            label = f"{label}（算不出来：{exc}）"
        passed = (obs <= limit) if op == "<=" else (obs >= limit)
        rows.append({"key": key, "label": label, "observed": obs,
                     "limit": limit, "op": op, "passed": bool(passed)})
    return {"n": len(rows),
            "n_passed": sum(1 for r in rows if r["passed"]),
            "n_failed": sum(1 for r in rows if not r["passed"]),
            "rows": rows}


# ===================================================================== 对抗
_CASES: list[dict[str, Any]] = []


def _case(name: str, claim: str) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        _CASES.append({"name": name, "claim": claim, "fn": fn})
        return fn
    return deco


# ------------------------------------------------------------- 配置 / spec
@_case("config_round_trips_through_disk",
       "配置**真的**从磁盘读回来（dump → load 逐位相同，不是比两次 `deep_merge`）")
def _c_cfg_round(p, c, ms, hs):
    import tempfile
    from .config import dump_config, load_config
    with tempfile.TemporaryDirectory(prefix="inf-cfg-") as td:
        tgt = Path(td) / "cfg.json"
        dump_config(c, tgt)
        back = load_config(tgt)
        ok = E.dumps(back) == E.dumps(c)
        return ok, ("逐位相同" if ok else "**不同**（磁盘往返改了配置）")


@_case("protocol_digest_is_stable",
       "协议指纹是口径的函数（**两份独立副本**算出来相同）")
def _c_digest_stable(p, c, ms, hs):
    """⚠️ 比的是**两份深拷贝**，不是「同一个对象算两次」。

    第一版写的是 `protocol_digest(c) == protocol_digest(c)` —— 那次变异测试
    当场抓出它是**弱的**：把 `protocol_digest` 换成 `lambda c: f"unstable-{id(c)}"`
    它**照样绿**（同一个对象两次调用 `id` 相同）。
    一份对象两次当然相等，这不叫「是口径的函数」，这叫「是内存地址的函数」。
    """
    import copy
    from .config import protocol_digest
    a = protocol_digest(c)
    b = protocol_digest(copy.deepcopy(c))
    # 再换一个**构造路径**（dump → load）拿第三份：口径相同就该指纹相同
    import tempfile
    from .config import dump_config, load_config
    with tempfile.TemporaryDirectory(prefix="inf-digest-") as td:
        tgt = Path(td) / "cfg.json"
        dump_config(c, tgt)
        d = protocol_digest(load_config(tgt))
    ok = a == b == d
    return ok, (f"{a}" if ok else f"两次不同：{a} / {b} / {d}")


@_case("protocol_digest_changes_when_the_axes_change",
       "协议指纹**真的吃**设计轴（改一根轴 ⇒ 指纹变）")
def _c_digest_eats_axes(p, c, ms, hs):
    """⚠️ 这条是上一条的**反向保险丝**：`protocol_digest_is_stable` 在
    指纹是个常数时也是绿的。两条一起才说明「指纹依赖口径且只依赖口径」。
    """
    import copy
    from .config import protocol_digest
    base = protocol_digest(c)
    changed = []
    for axis in ("backend", "quant", "batch", "context", "arrival"):
        alt = copy.deepcopy(c)
        alt["axes"][axis] = list(alt["axes"][axis]) + [alt["axes"][axis][0]]
        if protocol_digest(alt) != base:
            changed.append(axis)
    ok = len(changed) == 5
    return ok, f"改了这 5 根轴指纹都变：{changed}" if ok else f"只有 {changed} 会让指纹变"


@_case("axes_are_non_empty_and_have_no_duplicates",
       "每根设计轴非空、无重复（重复会让「324 条配置」这个数说谎）")
def _c_axes(p, c, ms, hs):
    bad = []
    for axis, vals in c["axes"].items():
        if not vals:
            bad.append(f"{axis} 为空")
        if len(set(map(str, vals))) != len(vals):
            bad.append(f"{axis} 有重复")
    return not bad, ("轴干净" if not bad else "；".join(bad))


@_case("every_backend_in_the_axis_is_declared_in_the_catalog",
       "设计轴里的每个后端都在目录里，每个量化都在 `quant` 里")
def _c_declared(p, c, ms, hs):
    bad = [b for b in c["axes"]["backend"] if b not in c["backends"]]
    bad += [q for q in c["axes"]["quant"] if q not in c["quant"]]
    return not bad, ("全部已声明" if not bad else "缺少：" + "、".join(bad))


# ------------------------------------------------------------- 矩阵
@_case("matrix_has_no_gap", "矩阵缺口为 0（全枚举，不是搜索）")
def _c_gap(p, c, ms, hs):
    m = p["matrix.json"]
    ok = m["matrix_gap"] == 0 and m["n_cells"] == m["expected_total"]
    return ok, f"{m['n_cells']}/{m['expected_total']}（缺口 {m['matrix_gap']}）"


@_case("matrix_has_no_duplicate_cells", "没有重复的格（同一部署单元 + 同一配置出现两次）")
def _c_dup(p, c, ms, hs):
    seen = {}
    dup = []
    for cell in _cells(p):
        k = (cell["deployment"], cell["config_id"])
        if k in seen:
            dup.append(str(k))
        seen[k] = 1
    return not dup, ("无重复" if not dup else f"{len(dup)} 个重复：{dup[:3]}")


@_case("every_cell_carries_a_reason", "每一格都有一个**非空**的原因字段")
def _c_reason(p, c, ms, hs):
    bad = [x["config_id"] for x in _cells(p)
           if not x.get("reason") or x["reason"] != ("ok" if x["feasible"] else x["reason"])]
    bad = [x["config_id"] for x in _cells(p) if not x.get("reason")]
    return not bad, (f"{len(_cells(p))} 格都有原因" if not bad else f"{len(bad)} 格没有")


@_case("infeasible_cells_carry_no_readings",
       "不可行的格子**不给读数**（给了就会被下游当成可用配置）")
def _c_no_read(p, c, ms, hs):
    bad = [x["config_id"] for x in _cells(p) if not x["feasible"] and x.get("readings")]
    return not bad, ("干净" if not bad else f"{len(bad)} 格不可行却带读数")


@_case("feasible_cells_have_every_reading",
       "可行的格子读数齐全且都是有限数（`inf` / `None` 不算读数）")
def _c_all_read(p, c, ms, hs):
    need = ("tokens_per_s", "p99_ms", "ttft_ms", "tpot_ms", "peak_bytes",
            "peak_frac", "rho", "bound_decode", "effective_batch")
    bad = []
    for x in _feas(p):
        r = x["readings"]
        for k in need:
            v = r.get(k)
            if v is None or (isinstance(v, float) and (v != v or v in
                                                       (float("inf"), float("-inf")))):
                bad.append(f"{x['config_id']}.{k}")
    return not bad, ("齐全" if not bad else f"{len(bad)} 处缺失：{bad[:3]}")


# ------------------------------------------------------------- 工作量 / roofline
@_case("analytic_and_layerwise_agree_on_flops",
       "闭式与逐层累加给出**同一个** flops（相对误差 ≤ 1e-12）")
def _c_flops(p, c, ms, hs):
    from . import cost as C
    prompt = int(c["workload"]["prompt_tokens"])
    bad = 0.0
    n = 0
    for m in ms:
        for bname, bspec in c["backends"].items():
            for qname, qspec in c["quant"].items():
                if qname not in bspec.get("supports_quant", []):
                    continue
                for batch in c["axes"]["batch"]:
                    for ctx in c["axes"]["context"]:
                        a1 = C.analytic_prefill(m, bspec, qspec, int(batch), prompt)
                        l1 = C.layerwise_prefill(m, bspec, qspec, int(batch), prompt)
                        a2 = C.analytic_decode(m, bspec, qspec, int(batch), int(ctx))
                        l2 = C.layerwise_decode(m, bspec, qspec, int(batch), int(ctx))
                        for a, l in ((a1, l1), (a2, l2)):
                            n += 1
                            bad = max(bad, _rel(a.flops, l.flops))
    return bad <= 1e-12, f"{n} 组，最大相对误差 {bad:.3e}"


@_case("analytic_and_layerwise_agree_on_bytes",
       "闭式与逐层累加给出**同一个**访存量（相对误差 ≤ 1e-12）")
def _c_bytes(p, c, ms, hs):
    from . import cost as C
    prompt = int(c["workload"]["prompt_tokens"])
    bad = 0.0
    n = 0
    for m in ms:
        for bname, bspec in c["backends"].items():
            for qname, qspec in c["quant"].items():
                if qname not in bspec.get("supports_quant", []):
                    continue
                for batch in c["axes"]["batch"]:
                    for ctx in c["axes"]["context"]:
                        a1 = C.analytic_prefill(m, bspec, qspec, int(batch), prompt)
                        l1 = C.layerwise_prefill(m, bspec, qspec, int(batch), prompt)
                        a2 = C.analytic_decode(m, bspec, qspec, int(batch), int(ctx))
                        l2 = C.layerwise_decode(m, bspec, qspec, int(batch), int(ctx))
                        for a, l in ((a1, l1), (a2, l2)):
                            n += 1
                            bad = max(bad, _rel(a.bytes_, l.bytes_))
    return bad <= 1e-12, f"{n} 组，最大相对误差 {bad:.3e}"


@_case("roofline_time_is_at_least_both_terms",
       "roofline 时间 ≥ 算力项且 ≥ 访存项（`max` 不是 `min`，也不是和）")
def _c_max(p, c, ms, hs):
    bad = []
    for x in _feas(p):
        for phase in ("prefill", "decode"):
            t = x["timing"][phase]
            if t["seconds"] + 1e-15 < max(t["compute_s"], t["memory_s"]):
                bad.append(f"{x['config_id']}.{phase}")
    return not bad, ("成立" if not bad else f"{len(bad)} 处违反：{bad[:3]}")


@_case("roofline_bound_names_the_larger_term",
       "`bound` 字段指的就是**更大**的那一项（标签与数值一致）")
def _c_bound(p, c, ms, hs):
    bad = []
    for x in _feas(p):
        for phase in ("prefill", "decode"):
            t = x["timing"][phase]
            want = "compute" if t["compute_s"] >= t["memory_s"] else "memory"
            if t["bound"] != want:
                bad.append(f"{x['config_id']}.{phase}")
    return not bad, ("一致" if not bad else f"{len(bad)} 处不一致：{bad[:3]}")


@_case("prefill_does_more_work_than_one_decode_step",
       "prefill 的算力 ≥ 单个 decode 步（它一次要算 `prompt` 个位置）")
def _c_pf_ge_dec(p, c, ms, hs):
    bad = [x["config_id"] for x in _feas(p)
           if x["readings"]["prefill_flops"] < x["readings"]["decode_flops"]]
    return not bad, ("成立" if not bad else f"{len(bad)} 处违反")


@_case("decode_flops_grow_with_context",
       "decode 的算力随 context **严格增长**（注意力要看更长的历史）")
def _c_flops_ctx(p, c, ms, hs):
    bad = []
    for key, grp in _groups(_feas(p), ("model", "hardware", "backend", "quant",
                                       "batch", "arrival_rate")).items():
        pts = sorted(grp, key=lambda x: int(x["context"]))
        for a, b in zip(pts, pts[1:]):
            if not b["readings"]["decode_flops"] > a["readings"]["decode_flops"]:
                bad.append(f"{key}@{a['context']}→{b['context']}")
    return not bad, ("严格增长" if not bad else f"{len(bad)} 处不增长：{bad[:3]}")


# ------------------------------------------------------------- 单调性
@_case("peak_memory_is_monotone_in_batch", "峰值占用随 batch 非降")
def _c_mem_batch(p, c, ms, hs):
    bad = []
    for key, grp in _groups(_cells(p), ("model", "hardware", "backend", "quant",
                                        "context", "arrival_rate")).items():
        pts = sorted(grp, key=lambda x: int(x["batch"]))
        for a, b in zip(pts, pts[1:]):
            if b["memory"]["peak_bytes"] + 1e-6 < a["memory"]["peak_bytes"]:
                bad.append(f"{key}@{a['batch']}→{b['batch']}")
    return not bad, ("非降" if not bad else f"{len(bad)} 处下降：{bad[:3]}")


@_case("peak_memory_is_monotone_in_context", "峰值占用随 context 非降")
def _c_mem_ctx(p, c, ms, hs):
    bad = []
    for key, grp in _groups(_cells(p), ("model", "hardware", "backend", "quant",
                                        "batch", "arrival_rate")).items():
        pts = sorted(grp, key=lambda x: int(x["context"]))
        for a, b in zip(pts, pts[1:]):
            if b["memory"]["peak_bytes"] + 1e-6 < a["memory"]["peak_bytes"]:
                bad.append(f"{key}@{a['context']}→{b['context']}")
    return not bad, ("非降" if not bad else f"{len(bad)} 处下降：{bad[:3]}")


@_case("kv_bytes_is_monotone_in_context", "KV 占用随 context 非降")
def _c_kv_ctx(p, c, ms, hs):
    bad = []
    for key, grp in _groups(_cells(p), ("model", "hardware", "backend", "quant",
                                        "batch", "arrival_rate")).items():
        pts = sorted(grp, key=lambda x: int(x["context"]))
        for a, b in zip(pts, pts[1:]):
            if b["memory"]["kv_bytes"] + 1e-6 < a["memory"]["kv_bytes"]:
                bad.append(f"{key}@{a['context']}→{b['context']}")
    return not bad, ("非降" if not bad else f"{len(bad)} 处下降：{bad[:3]}")


@_case("tpot_is_monotone_in_context", "TPOT 随 context 非降（KV 读得更多）")
def _c_tpot_ctx(p, c, ms, hs):
    bad = []
    for key, grp in _groups(_feas(p), ("model", "hardware", "backend", "quant",
                                       "batch", "arrival_rate")).items():
        pts = sorted(grp, key=lambda x: int(x["context"]))
        for a, b in zip(pts, pts[1:]):
            if b["readings"]["tpot_ms"] + 1e-12 < a["readings"]["tpot_ms"]:
                bad.append(f"{key}@{a['context']}→{b['context']}")
    return not bad, ("非降" if not bad else f"{len(bad)} 处下降：{bad[:3]}")


@_case("decode_step_time_is_monotone_in_batch",
       "单步时间随 batch 非降（一个 batch 里的请求共享权重，但 KV 读得更多）")
def _c_tpot_batch(p, c, ms, hs):
    bad = []
    for key, grp in _groups(_feas(p), ("model", "hardware", "backend", "quant",
                                       "context", "arrival_rate")).items():
        pts = sorted(grp, key=lambda x: int(x["batch"]))
        for a, b in zip(pts, pts[1:]):
            if b["readings"]["decode_s"] + 1e-12 < a["readings"]["decode_s"]:
                bad.append(f"{key}@{a['batch']}→{b['batch']}")
    return not bad, ("非降" if not bad else f"{len(bad)} 处下降：{bad[:3]}")


@_case("utilization_is_monotone_in_arrival", "利用率 ρ 随到达率非降")
def _c_rho(p, c, ms, hs):
    bad = []
    for key, grp in _groups(_cells(p), ("model", "hardware", "backend", "quant",
                                        "batch", "context")).items():
        pts = sorted(grp, key=lambda x: float(x["arrival_rate"]))
        vals = [x["readings"].get("rho") if x["feasible"] else float("inf")
                for x in pts]
        for a, b in zip(vals, vals[1:]):
            if b + 1e-12 < a:
                bad.append(str(key))
    return not bad, ("非降" if not bad else f"{len(bad)} 处下降")


@_case("p99_is_monotone_in_arrival", "P99 随到达率非降")
def _c_p99(p, c, ms, hs):
    bad = []
    for key, grp in _groups(_feas(p), ("model", "hardware", "backend", "quant",
                                       "batch", "context")).items():
        pts = sorted(grp, key=lambda x: float(x["arrival_rate"]))
        for a, b in zip(pts, pts[1:]):
            if b["readings"]["p99_ms"] + 1e-9 < a["readings"]["p99_ms"]:
                bad.append(f"{key}@{a['arrival_rate']}→{b['arrival_rate']}")
    return not bad, ("非降" if not bad else f"{len(bad)} 处下降：{bad[:3]}")


@_case("weight_bytes_is_monotone_in_quant_bits",
       "权重量越大体积越大（fp16 ≥ int8 ≥ int4）")
def _c_wbits(p, c, ms, hs):
    order = [int(v["weight_bits"]) for v in c["quant"].values()]
    bad = []
    for m in ms:
        prev = None
        for bits in sorted(set(order), reverse=True):
            cur = m.weight_bytes(bits)
            if prev is not None and cur > prev:
                bad.append(f"{m.name}@{bits}")
            prev = cur
    return not bad, ("单调" if not bad else "；".join(bad))


# ------------------------------------------------------------- 排队
@_case("erlang_b_is_in_the_unit_interval", "Erlang-B ∈ [0,1]")
def _c_eb(p, c, ms, hs):
    from . import queueing as Q
    bad = []
    for cc in (1, 2, 16, 256):
        for a in (0.0, 0.5, 1.0, 10.0, 200.0):
            v = Q.erlang_b(cc, a)
            if not (0.0 <= v <= 1.0):
                bad.append(f"c={cc},a={a}->{v}")
    return not bad, ("成立" if not bad else "；".join(bad[:3]))


@_case("erlang_c_is_at_least_erlang_b",
       "Erlang-C ≥ Erlang-B（要等的人 ⊇ 被阻塞的人）")
def _c_ec_ge_eb(p, c, ms, hs):
    from . import queueing as Q
    bad = []
    for cc in (1, 2, 8, 64):
        for a in (0.1, 1.0, 5.0, 30.0):
            if a / cc >= 1.0:
                continue
            if Q.erlang_c(cc, a) + 1e-12 < Q.erlang_b(cc, a):
                bad.append(f"c={cc},a={a}")
    return not bad, ("成立" if not bad else "；".join(bad[:3]))


@_case("unstable_cells_are_marked_infeasible", "ρ ≥ 1 的格子一律不可行")
def _c_unstable(p, c, ms, hs):
    bad = [x["config_id"] for x in _cells(p)
           if x["feasible"] and x["readings"]["rho"] >= 1.0]
    return not bad, ("成立" if not bad else f"{len(bad)} 格 ρ≥1 却可行")


@_case("p99_covers_at_least_the_service_time",
       "P99 ≥ 服务时间（排队只会让它更长，不会更短）")
def _c_p99_ge_s(p, c, ms, hs):
    bad = [x["config_id"] for x in _feas(p)
           if x["readings"]["p99_ms"] + 1e-9 < x["readings"]["service_s"] * 1e3]
    return not bad, ("成立" if not bad else f"{len(bad)} 处违反")


@_case("effective_batch_never_exceeds_the_configured_batch",
       "实际 batch ≤ 配置的 batch（不会因为排队而凭空变出服务台）")
def _c_eb_cap(p, c, ms, hs):
    bad = [x["config_id"] for x in _feas(p)
           if x["readings"]["effective_batch"] > int(x["batch"])]
    return not bad, ("成立" if not bad else f"{len(bad)} 处超出")


@_case("static_batching_wait_is_capped",
       "静态批的凑批等待被 timeout 封顶（不然低负载会算出几十秒的等待）")
def _c_wait_cap(p, c, ms, hs):
    cap = float(c["queue"]["max_batch_wait_s"])
    bad = [x["config_id"] for x in _feas(p)
           if x["readings"]["wait_batch_s"] > cap + 1e-12]
    return not bad, (f"≤ {cap}s" if not bad else f"{len(bad)} 处超过 {cap}s")


# ------------------------------------------------------------- 可行性
@_case("memory_feasibility_is_downward_closed_in_batch",
       "显存可行性在 batch 上**下闭**（非分页后端：B 装得下 ⇒ 更小的 B 也装得下）")
def _c_down_batch(p, c, ms, hs):
    bad = []
    for key, grp in _groups(_cells(p), ("model", "hardware", "backend", "quant",
                                        "context", "arrival_rate")).items():
        if any(x["reason"] in ("quant_unsupported", "no_gpu") for x in grp):
            continue
        if c["backends"][key[2]].get("continuous", False):
            continue          # 分页后端不受 `B·C` 约束
        pts = sorted(grp, key=lambda x: int(x["batch"]))
        ok_flags = [x["reason"] not in ("gpu_oom", "ram_oom") for x in pts]
        for i in range(len(ok_flags) - 1, -1, -1):
            if ok_flags[i] and any(not f for f in ok_flags[:i]):
                bad.append(str(key))
                break
    return not bad, ("下闭" if not bad else f"{len(bad)} 组不闭合：{bad[:3]}")


@_case("memory_feasibility_is_downward_closed_in_context",
       "显存可行性在 context 上**下闭**")
def _c_down_ctx(p, c, ms, hs):
    bad = []
    for key, grp in _groups(_cells(p), ("model", "hardware", "backend", "quant",
                                        "batch", "arrival_rate")).items():
        if any(x["reason"] in ("quant_unsupported", "no_gpu") for x in grp):
            continue
        if c["backends"][key[2]].get("continuous", False):
            continue
        pts = sorted(grp, key=lambda x: int(x["context"]))
        ok_flags = [x["reason"] not in ("gpu_oom", "ram_oom") for x in pts]
        for i in range(len(ok_flags) - 1, -1, -1):
            if ok_flags[i] and any(not f for f in ok_flags[:i]):
                bad.append(str(key))
                break
    return not bad, ("下闭" if not bad else f"{len(bad)} 组不闭合：{bad[:3]}")


@_case("paged_backends_do_not_reserve_the_whole_context",
       "分页后端按**实际用掉**的 token 分配，不做 `B·max_len` 预留")
def _c_paged(p, c, ms, hs):
    """⚠️ 这条守的是「vLLM 的 KV 占用与 context 无关」—— 正是它把
    「context 上限」与「batch 上限」解耦。若哪天把 `continuous` 的语义
    改回预留，这条会红。
    """
    used = int(c["workload"]["prompt_tokens"]) + int(c["workload"]["output_tokens"])
    bad = []
    for x in _cells(p):
        bspec = c["backends"][x["backend"]]
        if not bspec.get("continuous", False):
            continue
        want = min(used, int(x["context"])) * int(x["batch"])
        if x["memory"]["kv_tokens"] != want:
            bad.append(x["config_id"])
    return not bad, (f"分页后端只占 {used}·B 个槽" if not bad
                     else f"{len(bad)} 格预留了整段：{bad[:3]}")


@_case("no_gpu_hardware_rejects_gpu_backends", "没有 GPU 的机器上，GPU 后端一律不可行")
def _c_nogpu(p, c, ms, hs):
    bad = []
    for x in _cells(p):
        if x["hardware"] == "epyc-cpu-64g" and \
           c["backends"][x["backend"]]["device"] in ("gpu", "hybrid") and x["feasible"]:
            bad.append(x["config_id"])
    return not bad, ("成立" if not bad else f"{len(bad)} 格违反")


@_case("unsupported_quant_is_marked_not_silently_computed",
       "后端不支持的量化被标成 `quant_unsupported`，不参与读数")
def _c_qsup(p, c, ms, hs):
    bad = []
    for x in _cells(p):
        if x["quant"] not in c["backends"][x["backend"]].get("supports_quant", []):
            if x["feasible"] or x["reason"] != "quant_unsupported":
                bad.append(x["config_id"])
    return not bad, ("成立" if not bad else f"{len(bad)} 格违反")


@_case("oom_cells_really_exceed_the_device_budget",
       "报 OOM 的格子，占用**确实**超过了设备容量")
def _c_oom(p, c, ms, hs):
    bad = []
    for x in _cells(p):
        if x["reason"] not in ("gpu_oom", "ram_oom"):
            continue
        over = (x["memory"]["gpu_bytes"] > x["memory"]["budget_bytes"]
                if x["reason"] == "gpu_oom"
                else x["memory"]["ram_bytes"] > x["memory"]["budget_bytes"])
        if not over:
            bad.append(x["config_id"])
    return not bad, ("成立" if not bad else f"{len(bad)} 格误报 OOM")


@_case("feasible_cells_fit_the_device_budget", "可行格子的占用确实在容量之内")
def _c_fit(p, c, ms, hs):
    bad = []
    for x in _feas(p):
        m = x["memory"]
        if m["gpu_bytes"] > m["budget_bytes"] + 1e-6:
            bad.append(x["config_id"])
    return not bad, ("成立" if not bad else f"{len(bad)} 格超容量却可行")


# ------------------------------------------------------------- 决策
@_case("every_decision_is_a_feasible_cell", "每个部署决策都对应矩阵里一个**可行**的格")
def _c_dec_feas(p, c, ms, hs):
    by_id = {(x["deployment"], x["config_id"]): x for x in _cells(p)}
    bad = []
    for d in p["compare.json"]["per_deployment"]:
        dec = d.get("decision")
        if dec is None:
            continue
        cell = by_id.get((d["deployment"], dec["config_id"]))
        if cell is None or not cell["feasible"]:
            bad.append(d["deployment"])
        else:
            for k in ("tokens_per_s", "p99_ms", "tpot_ms"):
                if _rel(dec["readings"][k], cell["readings"][k]) > 1e-12:
                    bad.append(f"{d['deployment']}.{k}")
    return not bad, ("成立" if not bad else f"{len(bad)} 处不符")


@_case("every_decision_meets_the_slo", "每个部署决策都满足 SLO（P99 与 TPOT）")
def _c_dec_slo(p, c, ms, hs):
    bad = []
    for d in p["compare.json"]["per_deployment"]:
        dec = d.get("decision")
        if dec is None:
            continue
        r = dec["readings"]
        if r["p99_ms"] > float(c["slo"]["p99_ms"]) or \
           r["tpot_ms"] > float(c["slo"]["tpot_ms"]):
            bad.append(d["deployment"])
    return not bad, ("成立" if not bad else f"{len(bad)} 个决策超 SLO")


@_case("best_throughput_is_the_max_over_feasible_cells",
       "报出的「吞吐最优」确实是可行格里的最大值")
def _c_best_thr(p, c, ms, hs):
    bad = []
    for d in p["compare.json"]["per_deployment"]:
        bt = d.get("best_throughput")
        if bt is None:
            continue
        mine = [x for x in _feas(p) if x["deployment"] == d["deployment"]]
        top = max(x["readings"]["tokens_per_s"] for x in mine)
        if _rel(bt["readings"]["tokens_per_s"], top) > 1e-12:
            bad.append(d["deployment"])
    return not bad, ("成立" if not bad else f"{len(bad)} 个不符")


@_case("best_p99_is_the_min_over_feasible_cells", "报出的「P99 最优」确实是最小值")
def _c_best_p99(p, c, ms, hs):
    bad = []
    for d in p["compare.json"]["per_deployment"]:
        bp = d.get("best_p99")
        if bp is None:
            continue
        mine = [x for x in _feas(p) if x["deployment"] == d["deployment"]]
        lo = min(x["readings"]["p99_ms"] for x in mine)
        if _rel(bp["readings"]["p99_ms"], lo) > 1e-12:
            bad.append(d["deployment"])
    return not bad, ("成立" if not bad else f"{len(bad)} 个不符")


@_case("pareto_front_contains_no_dominated_cell",
       "前沿里没有任何一个被别的格支配（前沿就是前沿）")
def _c_pareto(p, c, ms, hs):
    from .compare import dominates
    bad = []
    for d in p["compare.json"]["per_deployment"]:
        mine = [x for x in _feas(p) if x["deployment"] == d["deployment"]]
        if not mine:
            continue
        front = [x for x in mine if x["config_id"] in set(d["pareto_ids"])]
        for a in front:
            if any(dominates(b, a, c) for b in mine if b is not a):
                bad.append(a["config_id"])
                break
        if len(front) != len(set(d["pareto_ids"])):
            bad.append(f"{d['deployment']}:前沿数与 id 数不符")
    return not bad, ("成立" if not bad else f"{len(bad)} 处：{bad[:3]}")


@_case("artifact_checksum_matches_its_own_body", "Artifact 的校验和 == 它自己 body 的摘要")
def _c_sum(p, c, ms, hs):
    import hashlib
    bad = []
    for a in p["artifact.json"]["artifacts"]:
        if a["status"] != "ok":
            continue
        want = hashlib.sha256(E.dumps(a["artifact"]).encode("utf-8")).hexdigest()[:16]
        if a["checksum"] != want:
            bad.append(a["deployment"])
    return not bad, ("一致" if not bad else f"{len(bad)} 个不符")


@_case("deploy_rows_recheck_the_slo_against_the_artifact",
       "Deploy 步的 SLO 判定与 Artifact 里的读数一致（不是另判一套）")
def _c_deploy(p, c, ms, hs):
    bad = []
    for r in p["deploy.json"]["rows"]:
        if r["status"] != "ok":
            continue
        want = (r["p99_ms"] <= float(c["slo"]["p99_ms"])
                and r["tpot_ms"] <= float(c["slo"]["tpot_ms"]))
        if bool(r["slo_met"]) != want:
            bad.append(r["deployment"])
    return not bad, ("一致" if not bad else f"{len(bad)} 处不一致")


# ------------------------------------------------------------- 曲线
@_case("every_series_has_all_three_curves", "每条序列都有 batch / context / arrival 三条曲线")
def _c_curves(p, c, ms, hs):
    bad = [s["series_id"] for s in _series(p)
           if set(s["curves"]) != {"batch", "context", "arrival"}]
    return not bad, ("齐全" if not bad else f"{len(bad)} 条缺曲线")


@_case("curve_axes_are_strictly_increasing", "曲线的 x 轴严格递增（画出来才不会打结）")
def _c_curve_x(p, c, ms, hs):
    bad = []
    for s in _series(p):
        for ax, pts in s["curves"].items():
            xs = [q["x"] for q in pts]
            if any(b <= a for a, b in zip(xs, xs[1:])):
                bad.append(f"{s['series_id']}.{ax}")
    return not bad, ("递增" if not bad else f"{len(bad)} 条不齐：{bad[:3]}")


@_case("scaling_point_count_matches_the_declared_grids",
       "曲线点数 == 轴网格长度（少一个点就是静默跳过了某些 x）")
def _c_curve_n(p, c, ms, hs):
    want = {ax: len(meta["grid"]) for ax, meta in p["scaling.json"]["axes"].items()}
    bad = []
    for s in _series(p):
        for ax, pts in s["curves"].items():
            if len(pts) != want[ax]:
                bad.append(f"{s['series_id']}.{ax}")
    return not bad, (f"合计 {p['scaling.json']['n_points']} 点" if not bad
                     else f"{len(bad)} 条点数不符：{bad[:3]}")


# ------------------------------------------------------------- 纯度 / 确定性
@_case("no_undeclared_time_key_in_the_versioned_products",
       "版本化产物里的**时间类**键全在白名单里（白名单 = 模型预测，不是实测）")
def _c_time_keys(p, c, ms, hs):
    """⚠️ 判据不是「没有时间键」（那不可能，这是个性能项目），
    而是「**每一个**时间键都被声明过」。于是有人往产物里塞一个实测耗时，
    这条会红 —— 而「实测」混进「预测」正是这类数字孪生最危险的失真。
    """
    bad: list[str] = []
    seen: set[str] = set()

    def walk(obj: Any, path: str, exempt: bool) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k)
                here = f"{path}.{key}" if path else key
                # ⚠️ `str.endswith` 直接吃元组：原先对 4 个后缀逐个 `any(...)`，
                #    在 1944 格的矩阵上会多做 ~37 万次函数调用。
                if key.endswith(_TIME_SUFFIXES):
                    seen.add(key)
                    if not exempt and key not in _MODEL_TIME_KEYS:
                        bad.append(here)
                walk(v, here, exempt)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]", exempt)

    for name in E.VERSIONED_FILES:
        if name not in p:
            continue
        # 目录类产物（`config.json` / `backends.json`）是**口径本身**：
        # 里面的「时间」是参数（超时、launch 开销），不是这台机器测出来的。
        walk(p[name], name, exempt=(name in _CATALOG_FILES))
    extra = sorted(seen - _MODEL_TIME_KEYS)
    ok = not bad
    note = f"{len(seen)} 个时间类键全在白名单内" if ok else f"未声明：{bad[:5]}"
    if extra:
        note += f"；白名单外但被豁免（config.json）：{extra[:6]}"
    return ok, note


@_case("version_digest_covers_every_declared_file",
       "`version.json` 声明的文件列表 == `VERSIONED_FILES`，且都真的存在")
def _c_version(p, c, ms, hs):
    v = p.get("version.json", {})
    decl = list(v.get("versioned_files", []))
    same = decl == list(E.VERSIONED_FILES)
    missing = [n for n in E.VERSIONED_FILES if n not in p]
    ok = same and not missing
    return ok, (f"{len(decl)} 个文件全部在位" if ok
                else f"声明不符={not same}；缺失={missing}")


@_case("renderers_are_pure_functions_of_the_products",
       "报告只由产物决定（同产物 ⇒ 同报告；JSON 往返后仍相同）")
def _c_pure(p, c, ms, hs):
    """⚠️ 为什么要比 **JSON 往返副本**：渲染器若依赖 dict 的**插入顺序**，
    直接渲染两次是一样的（顺序没变），但盘上产物的键序是字典序 ——
    于是 `visualize`（从盘上读）会报「重建的报告与盘上的不同」，
    而 `run` 是绿的。这条用例必须**自己造出那个键序**才抓得到。
    """
    from .report import render
    a_md, a_html = render(p)
    b_md, b_html = render(p)
    rt = json.loads(json.dumps(p, ensure_ascii=False, sort_keys=True))
    c_md, c_html = render(rt)
    ok = (a_md == b_md == c_md) and (a_html == b_html == c_html)
    return ok, ("纯净" if ok else "**报告依赖了产物之外的东西或键序**")


@_case("adversarial_cases_all_ran_with_their_inputs_present",
       "对抗用例跑的时候，**它声明要读的产物**都在（缺输入不能被兜成「违反」）")
def _c_inputs(p, c, ms, hs):
    missing = [n for n in ADVERSARIAL_INPUTS if n not in p]
    return not missing, (f"{len(ADVERSARIAL_INPUTS)} 份输入都在位" if not missing
                         else "缺少：" + "、".join(missing))


#: 需要**完整产物**（含 `summary.json`）才能跑的用例 —— 它们要渲报告，
#: 而报告里要印对抗结论 ⇒ 环。拆法：对抗层跑**两趟**（见 `pipeline`）。
REPORT_DEPENDENT_CASES: frozenset[str] = frozenset({
    "renderers_are_pure_functions_of_the_products",
    "adversarial_cases_all_ran_with_their_inputs_present",
    # ⚠️ 这条**也要**在第二趟：第一趟时 `config.json` / `summary.json` /
    #    `adversarial.json` 还没生成，它比对的是「声明 vs 实际在位」，
    #    而产物集本身在第二趟才完整。放第一趟 ⇒ 它恒红，
    #    而恒红的判据会被误读成「版本号坏了」。
    "version_digest_covers_every_declared_file",
})

#: 对抗用例**真正读**的产物（不含 `summary.json` —— 第一趟时它还没生成）。
ADVERSARIAL_INPUTS: tuple[str, ...] = (
    "models.json", "hardware.json", "backends.json",
    "inspect.json", "selection.json", "convert.json", "warmup.json",
    "matrix.json", "stress.json", "scaling.json", "compare.json",
    "artifact.json", "deploy.json", "crosscheck.json",
    "acceptance.json", "expectations.json", "config.json", "version.json",
)


def run_adversarial(products: dict[str, Any], cfg: dict[str, Any],
                    models: tuple[ModelSpec, ...],
                    hardware: tuple[HardwareSpec, ...],
                    only: frozenset[str] | None = None,
                    exclude: frozenset[str] | None = None) -> dict[str, Any]:
    rows = []
    for spec in _CASES:
        name = spec["name"]
        if only is not None and name not in only:
            continue
        if exclude is not None and name in exclude:
            continue
        try:
            held, detail = spec["fn"](products, cfg, models, hardware)
        except Exception as exc:                        # noqa: BLE001
            held, detail = False, f"**抛异常**：{type(exc).__name__}: {exc}"
        rows.append({"name": name, "claim": spec["claim"],
                     "held": bool(held), "detail": str(detail)})
    return {"n_cases": len(rows),
            "n_held": sum(1 for r in rows if r["held"]),
            "n_violated": sum(1 for r in rows if not r["held"]),
            "cases": rows,
            "inputs_present": [n for n in ADVERSARIAL_INPUTS if n in products]}


def merge_adversarial(*parts: dict[str, Any]) -> dict[str, Any]:
    """把**两趟**对抗的结果合成一份（`REPORT_DEPENDENT_CASES` 在第二趟）。

    ⚠️ 顺序要按 `_CASES` 的声明序，不是「先跑的在前」——
    否则两次 `run` 的报告会因为合并顺序不同而不同。
    """
    by_name: dict[str, dict[str, Any]] = {}
    for part in parts:
        for r in part.get("cases", []):
            by_name[r["name"]] = r
    rows = [by_name[s["name"]] for s in _CASES if s["name"] in by_name]
    return {"n_cases": len(rows),
            "n_held": sum(1 for r in rows if r["held"]),
            "n_violated": sum(1 for r in rows if not r["held"]),
            "cases": rows,
            "inputs_present": sorted({n for p in parts
                                      for n in p.get("inputs_present", [])})}
