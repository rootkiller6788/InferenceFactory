"""收拢：**可行域 → Pareto 前沿 → 部署决策 → Artifact**。

扇出-收拢型的「收拢」这一半。三条：

1. **可行域是产物**：`B·C ≤ K` 那条双曲线的 `K` 要算出来（它依赖后端是否分页）。
2. **决策不是「Pareto 里挑一个好看的」**，而是「**在 SLO 内最大化吞吐**」——
   SLO 是硬门槛，不是第四个目标。混进目标里会让「吞吐最高的那格」被
   一个延迟很差的格子挡住，而那格本来就不该进候选。
3. **「吞吐最优」与「满足 SLO」不是同一个点**（实测确实不是）——
   这条要单独报，因为它是「曲线 > 单点」的直接证据。
"""

from __future__ import annotations

import hashlib
from typing import Any

from . import emit as E
from . import memory as M
from .specs import HardwareSpec, ModelSpec
from .space import deployment_id


def _obj_keys(cfg: dict[str, Any]) -> tuple[str, ...]:
    return tuple(cfg["objectives"].keys())


def _better(a: dict[str, Any], b: dict[str, Any], cfg: dict[str, Any]) -> int:
    """a 相对 b 在三目标上的比较：1 = 更好 / -1 = 更差 / 0 = 完全相等。"""
    gt = lt = 0
    for name, spec in cfg["objectives"].items():
        k = spec["key"]
        va, vb = a["readings"][k], b["readings"][k]
        if va > vb:
            gt += 1 if spec["direction"] > 0 else -1
        elif va < vb:
            lt += 1 if spec["direction"] > 0 else -1
    better = gt - lt
    if better > 0:
        return 1
    if better < 0:
        return -1
    return 0


def dominates(a: dict[str, Any], b: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """a 支配 b：所有目标不更差，且至少一个严格更好。"""
    objs = cfg["objectives"]
    strictly = False
    for name, spec in objs.items():
        k = spec["key"]
        va, vb = a["readings"][k], b["readings"][k]
        d = spec["direction"]
        if va * d > vb * d:
            strictly = True
        elif va * d < vb * d:
            return False
    return strictly


def pareto_front(cells: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """前沿：**保留**并列项（同一组目标值的不同配置是不同决策，都要留）。

    ⚠️ 前沿按 `readings` 去重会让「同一性能、不同 VRAM」这种真正不同的
    权衡点被合并掉 —— 数量好看了，信息没了。
    """
    keep: list[dict[str, Any]] = []
    for a in cells:
        if any(dominates(b, a, cfg) for b in cells if b is not a):
            continue
        keep.append(a)
    return keep


def _slo_ok(cell: dict[str, Any], cfg: dict[str, Any]) -> bool:
    slo = cfg["slo"]
    r = cell["readings"]
    return bool(r["p99_ms"] <= float(slo["p99_ms"])
                and r["tpot_ms"] <= float(slo["tpot_ms"]))


def _rank_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    """决策的排序键：吞吐最大 → P99 最小 → 占用最小 → id 最小。

    ⚠️ tie-break 必须走到**全部**目标 + id，否则「并列」时选到哪个
    取决于 `min` 的稳定性之外的东西（比如列表顺序）。
    """
    r = cell["readings"]
    return (-r["tokens_per_s"], r["p99_ms"], r["peak_bytes"], cell["config_id"])


def hyperbola_of(model: ModelSpec, hw: HardwareSpec, backend: dict[str, Any],
                 quant: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """`B·C ≤ K` 里的 K —— **可行域的双曲线**。

    ⚠️ 分页后端（`continuous`）**没有**这条双曲线：它只按实际用掉的 token
    分配 ⇒ `B·(prompt+output) ≤ K`，与 `C` 无关。
    「context 上限与 batch 上限解耦」就是 vLLM 那套分页 KV 的定量表达。
    """
    prompt = int(cfg["workload"]["prompt_tokens"])
    output = int(cfg["workload"]["output_tokens"])
    kv_bits = int(quant["kv_bits"])
    kv_per_token = model.kv_bytes_per_token(kv_bits)
    util = float(backend["kv_util"])
    paged = bool(backend.get("continuous", False))
    # 预算里除了权重/框架，还有 prefill 的激活（它跟 batch 走）
    budget0 = M.kv_budget_bytes(model, backend, quant, hw, 1, 1, prompt, output)
    return {
        "paged": paged,
        "kv_per_token": kv_per_token,
        "kv_util": util,
        "budget_bytes_at_batch1": budget0,
        # 非分页：`kv_tokens = B·C` ⇒ `B·C ≤ budget·util/kv_per_token`
        "max_bc_tokens": budget0 * util / kv_per_token if kv_per_token > 0 else float("inf"),
        # 分页：`kv_tokens = B·(prompt+output)` ⇒ 与 C 无关
        "max_batch_tokens": budget0 * util / kv_per_token if kv_per_token > 0 else float("inf"),
        "applies": not paged,
    }


def compare_all(matrix: dict[str, Any], models: tuple[ModelSpec, ...],
                hardware: tuple[HardwareSpec, ...],
                cfg: dict[str, Any]) -> dict[str, Any]:
    """收拢：**每个部署单元出一份决策**。"""
    cells = matrix["cells"]
    reasons: dict[str, int] = {}
    for c in cells:
        if not c["feasible"]:
            reasons[c["reason"]] = reasons.get(c["reason"], 0) + 1

    per: list[dict[str, Any]] = []
    winners: set[str] = set()
    n_slo_exclusions = 0
    for m in models:
        for hw in hardware:
            did = deployment_id(m, hw)
            mine = [c for c in cells
                    if c["model"] == m.name and c["hardware"] == hw.name]
            feas = [c for c in mine if c["feasible"]]
            if not feas:
                # ⚠️ `slo_blocked` 也要给：下游 `package_artifacts` 读它，
                #    而「没有可行格」与「有可行格但都过不了 SLO」是**两种**失败。
                #    少给这个键 ⇒ `KeyError`，看起来像打包坏了，其实是 compare 漏了字段。
                per.append({"deployment": did, "n_cells": len(mine),
                            "n_feasible": 0, "n_pareto": 0, "pareto_ids": [],
                            "decision": None, "reason": "no_feasible_cell",
                            "slo_blocked": False, "n_slo_ok": 0,
                            "throughput_opt_is_slo_ok": False,
                            "best_throughput": None, "best_p99": None,
                            "best_footprint": None, "hyperbolas": {}})
                continue

            front = pareto_front(feas, cfg)
            best_thr = min(feas, key=_rank_key)
            best_p99 = min(feas, key=lambda c: (c["readings"]["p99_ms"],
                                                c["config_id"]))
            best_foot = min(feas, key=lambda c: (c["readings"]["peak_bytes"],
                                                 c["config_id"]))
            slo_ok = [c for c in feas if _slo_ok(c, cfg)]
            decision = min(slo_ok, key=_rank_key) if slo_ok else None
            blocked = decision is None
            if blocked:
                n_slo_exclusions += 1
            elif not _slo_ok(best_thr, cfg):
                # 吞吐最优的那格**本身**不满足 SLO ⇒ 这也是一次「被 SLO 淘汰」
                n_slo_exclusions += 1
            if decision is not None:
                winners.add(decision["backend"])

            # ---- 可行域的双曲线（按后端 × 量化）----
            hyps: dict[str, Any] = {}
            for bname, bspec in cfg["backends"].items():
                for qname in cfg["axes"]["quant"]:
                    if qname not in bspec.get("supports_quant", []):
                        continue
                    hyps[f"{bname}|{qname}"] = hyperbola_of(
                        m, hw, bspec, cfg["quant"][qname], cfg)

            per.append({
                "deployment": did,
                "n_cells": len(mine),
                "n_feasible": len(feas),
                "n_pareto": len(front),
                "pareto_ids": sorted(c["config_id"] for c in front),
                "best_throughput": _slim(best_thr),
                "best_p99": _slim(best_p99),
                "best_footprint": _slim(best_foot),
                "n_slo_ok": len(slo_ok),
                "decision": _slim(decision) if decision else None,
                "slo_blocked": blocked,
                "throughput_opt_is_slo_ok": _slo_ok(best_thr, cfg),
                "hyperbolas": hyps,
            })

    decisions = {p["deployment"]: (p["decision"] or {}).get("backend")
                 for p in per}
    return {
        "n_cells": len(cells),
        "n_feasible": sum(1 for c in cells if c["feasible"]),
        "infeasible_reasons": dict(sorted(reasons.items())),
        "n_infeasible_reasons": len(reasons),
        "per_deployment": per,
        "n_backend_wins": len(winners),
        "backend_wins": sorted(winners),
        "n_slo_exclusions": n_slo_exclusions,
        "decision_by_deployment": decisions,
        "best_strategy_is_shared": len(set(
            (p["decision"] or {}).get("config_id") for p in per)) == 1,
    }


def _slim(c: dict[str, Any] | None) -> dict[str, Any] | None:
    if c is None:
        return None
    return {
        "config_id": c["config_id"],
        "backend": c["backend"], "quant": c["quant"],
        "batch": c["batch"], "context": c["context"],
        "arrival_rate": c["arrival_rate"],
        "effective_batch": c["effective_batch"],
        "readings": {k: c["readings"][k] for k in
                     ("tokens_per_s", "p99_ms", "ttft_ms", "tpot_ms",
                      "peak_bytes", "peak_frac", "rho", "bound_decode")},
    }


def package_artifacts(compare: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """第 9 步 Package：把决策冻成一个**带校验和**的可部署声明。

    ⚠️ 校验和吃的是**规范化 JSON**（`sort_keys`）—— 吃 dict 的话，
    换一次插入顺序就换一次校验和，而「artifact 变了没有」这个问题
    就再也不能被校验和回答了。
    """
    arts = []
    for p in compare["per_deployment"]:
        d = p["decision"]
        if d is None:
            arts.append({"deployment": p["deployment"], "status": "no_decision",
                         "reason": "slo_blocked" if p["slo_blocked"]
                         else "no_feasible_cell"})
            continue
        body = {
            "deployment": p["deployment"],
            "backend": d["backend"],
            "quant": d["quant"],
            "serving": {"max_batch": d["batch"], "max_context": d["context"],
                        "target_arrival_rate": d["arrival_rate"]},
            "slo": dict(cfg["slo"]),
            "expected": d["readings"],
        }
        blob = E.dumps(body)
        arts.append({"deployment": p["deployment"], "status": "ok",
                     "config_id": d["config_id"],
                     "artifact": body,
                     "checksum": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]})
    return {"n_artifacts": len(arts),
            "n_ok": sum(1 for a in arts if a["status"] == "ok"),
            "artifacts": arts}


def deploy_check(artifacts: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """第 10 步 Deploy：拿**决策时的读数**复核 SLO，并给回滚条件。

    ⚠️ 这里不做新的测量（本机没有 GPU），复核的是「这套声明在它自己
    的模型口径下是否兑现 SLO」。把「复核」说成「验收」会让人以为测过了。
    """
    rows = []
    for a in artifacts["artifacts"]:
        if a["status"] != "ok":
            rows.append({"deployment": a["deployment"], "status": a["status"],
                         "slo_met": False, "note": "无决策，不部署"})
            continue
        exp = a["artifact"]["expected"]
        slo = cfg["slo"]
        met = (exp["p99_ms"] <= float(slo["p99_ms"])
               and exp["tpot_ms"] <= float(slo["tpot_ms"]))
        rows.append({
            "deployment": a["deployment"],
            "status": "ok",
            "config_id": a["config_id"],
            "checksum": a["checksum"],
            "slo_met": met,
            "p99_ms": exp["p99_ms"],
            "tpot_ms": exp["tpot_ms"],
            "tokens_per_s": exp["tokens_per_s"],
            "rollback_if": f"P99 > {slo['p99_ms']:g} ms 或 "
                           f"TPOT > {slo['tpot_ms']:g} ms",
        })
    return {"n_rows": len(rows),
            "n_slo_met": sum(1 for r in rows if r.get("slo_met")),
            "rows": rows}
