"""主流水线十步 + 三层核对 + 写盘。

十步：Inspect → Select Backend → Convert/Load → Warmup → Benchmark →
Stress Test → Profile → Compare → Package → Deploy。
**每一步都有一个产物** —— 「每一级 IR 都是产物」这条在
`L6-tensorcompilerfactory` 里买到的东西，这里照旧：只留最终决策的话，
「为什么选了它」就没有可审计的对象。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import compare as CMP
from . import cost as C
from . import emit as E
from . import engine as ENG
from . import expectations as EXP
from . import invariants as INV
from . import memory as M
from . import queueing as Q
from . import report as R
from . import scaling as SC
from . import space as SP
from . import visualize as VIS
from .config import protocol_digest
from .specs import HardwareSpec, ModelSpec, load_hardware, load_models

_PKG_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Run:
    products: dict[str, Any] = field(default_factory=dict)
    out_dir: Path = Path("out")
    timings: dict[str, float] = field(default_factory=dict)


def _tick(t0: float, sink: dict[str, float], key: str) -> float:
    now = time.perf_counter()
    sink[key] = sink.get(key, 0.0) + (now - t0)
    return now


# --------------------------------------------------------------------- 步 1–4
def _binding_budget(hw: HardwareSpec) -> float:
    return hw.gpu_usable_bytes if hw.has_gpu else hw.ram_bytes


def step_inspect(models: tuple[ModelSpec, ...],
                 hardware: tuple[HardwareSpec, ...],
                 cfg: dict[str, Any]) -> dict[str, Any]:
    """步 1 Inspect：模型到底有多大、KV 有多贵、**权重放不放得下**。"""
    rows = []
    for m in models:
        for hw in hardware:
            budget = _binding_budget(hw)
            rows.append({
                "deployment": SP.deployment_id(m, hw),
                "model": m.name, "hardware": hw.name,
                "params": m.params,
                "params_b": m.params / 1e9,
                "n_layers": m.n_layers, "hidden": m.hidden,
                "gqa_ratio": m.n_heads / m.n_kv_heads,
                "weight_bytes": {q: m.weight_bytes(int(v["weight_bits"]))
                                 for q, v in cfg["quant"].items()},
                "kv_bytes_per_token": {q: m.kv_bytes_per_token(int(v["kv_bits"]))
                                       for q, v in cfg["quant"].items()},
                "budget_bytes": budget,
                # ⚠️ 「权重放不放得下」是**算出来的**：`params·bits/8 > 预算`
                "weights_fit": {q: m.weight_bytes(int(v["weight_bits"])) <= budget
                                for q, v in cfg["quant"].items()},
            })
    return {"n_deployments": len(rows), "rows": rows}


def step_selection(models: tuple[ModelSpec, ...],
                   hardware: tuple[HardwareSpec, ...],
                   cfg: dict[str, Any]) -> dict[str, Any]:
    """步 2 Select Backend：按**设备**与**权重体积**筛掉不可能的后端。

    ⚠️ 筛掉的原因要单独留着 —— 「这个后端没被选」有两种完全不同的理由：
    机器没有那个设备（**永远不可能**），和权重放不下（**换个量化就行**）。
    合成一个 `False` 的话，「量化解决了什么」这个问题就答不了。
    """
    rows = []
    for m in models:
        for hw in hardware:
            budget = _binding_budget(hw)
            for bname, bspec in cfg["backends"].items():
                want_gpu = str(bspec["device"]) in ("gpu", "hybrid")
                device_ok = (not want_gpu) or hw.has_gpu
                fit = {}
                for q, v in cfg["quant"].items():
                    if q not in bspec.get("supports_quant", []):
                        fit[q] = False
                        continue
                    wb = m.weight_bytes(int(v["weight_bits"]))
                    off = float(bspec.get("offload", 1.0)) if want_gpu else 1.0
                    fit[q] = (wb * off) <= budget
                viable = device_ok and any(fit.values())
                rows.append({
                    "deployment": SP.deployment_id(m, hw),
                    "backend": bname, "device": bspec["device"],
                    "device_ok": device_ok,
                    "quants_supported": list(bspec.get("supports_quant", [])),
                    "quants_that_fit": sorted(q for q, ok in fit.items() if ok),
                    "viable": viable,
                    "block_reason": (None if viable else
                                     "no_gpu" if not device_ok else "weights_too_big"),
                })
    return {"n_rows": len(rows),
            "n_viable": sum(1 for r in rows if r["viable"]),
            "rows": rows}


def step_convert(models: tuple[ModelSpec, ...],
                 hardware: tuple[HardwareSpec, ...],
                 cfg: dict[str, Any]) -> dict[str, Any]:
    """步 3 Convert / Load：量化转换。

    ⚠️ `convert_seconds` 是**模型预测**（按设备的写带宽写一遍目标体积），
    不是实测 —— 本机没有 GPU，也没有真的权重文件。
    """
    rows = []
    for m in models:
        for hw in hardware:
            bw = hw.gpu_bw if hw.has_gpu else hw.ram_bw
            for bname, bspec in cfg["backends"].items():
                for qname, qspec in cfg["quant"].items():
                    if qname not in bspec.get("supports_quant", []):
                        continue
                    src = m.weight_bytes(16)
                    dst = m.weight_bytes(int(qspec["weight_bits"]))
                    rows.append({
                        "deployment": SP.deployment_id(m, hw),
                        "backend": bname, "quant": qname,
                        "source_bytes": src, "target_bytes": dst,
                        "reduction": dst / src if src else 0.0,
                        "bytes_written": dst,
                        "convert_seconds": dst / bw if bw else float("inf"),
                    })
    return {"n_rows": len(rows), "rows": rows}


def step_warmup(models: tuple[ModelSpec, ...],
                hardware: tuple[HardwareSpec, ...],
                cfg: dict[str, Any]) -> dict[str, Any]:
    """步 4 Warmup：预热后**常驻**多少、预热多久。

    ⚠️ 常驻 = 权重 + 框架（**不含** KV 与激活）—— 它是「什么都不跑」时的
    基线，也是「这个后端还能剩多少给 KV」的起点。
    """
    rows = []
    for m in models:
        for hw in hardware:
            bw = hw.gpu_bw if hw.has_gpu else hw.ram_bw
            for bname, bspec in cfg["backends"].items():
                for qname, qspec in cfg["quant"].items():
                    if qname not in bspec.get("supports_quant", []):
                        continue
                    wb = m.weight_bytes(int(qspec["weight_bits"]))
                    fw = float(bspec["framework_bytes"])
                    resident = wb + fw
                    rows.append({
                        "deployment": SP.deployment_id(m, hw),
                        "backend": bname, "quant": qname,
                        "weight_bytes": wb, "framework_bytes": fw,
                        "resident_bytes": resident,
                        # 预热要把权重读两遍：一次加载、一次首次推理（cache 冷）
                        "warmup_seconds": 2.0 * wb / bw if bw else float("inf"),
                        "resident_frac": resident / _binding_budget(hw),
                    })
    return {"n_rows": len(rows), "rows": rows}


# --------------------------------------------------------------------- 步 5–6
def step_matrix(models: tuple[ModelSpec, ...],
                hardware: tuple[HardwareSpec, ...],
                cfg: dict[str, Any]) -> dict[str, Any]:
    """步 5 Benchmark：**全矩阵**（笛卡尔积，缺口必须为 0）。"""
    prompt = int(cfg["workload"]["prompt_tokens"])
    output = int(cfg["workload"]["output_tokens"])
    wait_cap = float(cfg["queue"]["max_batch_wait_s"])
    configs = SP.enumerate_configs(cfg)
    cells: list[dict[str, Any]] = []
    for m in models:
        for hw in hardware:
            for c in configs:
                bname = c["backend"]
                qname = c["quant"]
                bspec = cfg["backends"][bname]
                qspec = cfg["quant"][qname]
                cell = ENG.evaluate(m, bname, bspec, qname, qspec, hw,
                                    int(c["batch"]), int(c["context"]),
                                    float(c["arrival"]), prompt, output)
                cell["config_id"] = SP.config_id(c)
                cell["describe"] = SP.describe(c)
                cell["deployment"] = SP.deployment_id(m, hw)
                cells.append(cell)

    reasons: dict[str, int] = {}
    for c in cells:
        if not c["feasible"]:
            reasons[c["reason"]] = reasons.get(c["reason"], 0) + 1
    expected = SP.matrix_expected(models, hardware, cfg)
    return {
        "n_models": len(models), "n_hardware": len(hardware),
        "n_configs": len(configs),
        "expected_total": expected,
        "n_cells": len(cells),
        "matrix_gap": expected - len(cells),
        "n_feasible": sum(1 for c in cells if c["feasible"]),
        "infeasible_reasons": dict(sorted(reasons.items())),
        "axes": {k: list(v) for k, v in cfg["axes"].items()},
        "cells": cells,
    }


def step_stress(matrix: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """步 6 Stress Test：**能扛住多大到达率**。

    ⚠️ 「扛不住」有两种，必须分开报：
    - `unstable`（ρ ≥ 1）：队列会无限增长 —— 是**排队**问题，加显存没用
    - `*_oom`：显存放不下 —— 是**容量**问题，排队再好也没用
    合成一个「不行」的话，「扩容还是换后端」这个问题就答不了。
    """
    arrivals = sorted(cfg["axes"]["arrival"], key=float)
    per: dict[str, dict[str, Any]] = {}
    for c in matrix["cells"]:
        d = per.setdefault(c["deployment"], {
            "deployment": c["deployment"], "n_cells": 0,
            "by_arrival": {str(a): {"feasible": 0, "unstable": 0, "oom": 0,
                                    "other": 0} for a in arrivals},
            "by_backend": {},
        })
        d["n_cells"] += 1
        a = str(float(c["arrival_rate"]))
        bucket = d["by_arrival"].setdefault(a, {"feasible": 0, "unstable": 0,
                                                "oom": 0, "other": 0})
        if c["feasible"]:
            bucket["feasible"] += 1
        elif c["reason"] in ("gpu_oom", "ram_oom"):
            bucket["oom"] += 1
        elif c["reason"] == "unstable":
            bucket["unstable"] += 1
        else:
            bucket["other"] += 1
        bb = d["by_backend"].setdefault(c["backend"],
                                        {"feasible": 0, "unstable": 0, "oom": 0,
                                         "other": 0, "max_arrival": None})
        if c["feasible"]:
            bb["feasible"] += 1
            cur = float(c["arrival_rate"])
            if bb["max_arrival"] is None or cur > bb["max_arrival"]:
                bb["max_arrival"] = cur
        elif c["reason"] in ("gpu_oom", "ram_oom"):
            bb["oom"] += 1
        elif c["reason"] == "unstable":
            bb["unstable"] += 1
        else:
            bb["other"] += 1

    rows = []
    for d in sorted(per.values(), key=lambda x: x["deployment"]):
        best = None
        for bb in d["by_backend"].values():
            if bb["max_arrival"] is not None and (best is None
                                                  or bb["max_arrival"] > best):
                best = bb["max_arrival"]
        rows.append({**d, "max_arrival_feasible": best})
    return {"n_deployments": len(rows), "arrivals": [str(a) for a in arrivals],
            "rows": rows}


# --------------------------------------------------------------------- 互校
def build_crosscheck(models: tuple[ModelSpec, ...],
                     hardware: tuple[HardwareSpec, ...],
                     scaling: dict[str, Any],
                     cfg: dict[str, Any]) -> dict[str, Any]:
    """**双实现互校**：解析 vs 逐层（工作量）、Erlang-C vs 离散事件（排队）。"""
    prompt = int(cfg["workload"]["prompt_tokens"])
    output = int(cfg["workload"]["output_tokens"])
    rel = lambda a, b: (abs(a - b) / abs(b)) if b else (0.0 if a == 0 else float("inf"))

    n_f = n_f_ok = 0
    max_f = 0.0
    n_b = n_b_ok = 0
    max_b = 0.0
    for m in models:
        for hw in hardware:
            for bname, bspec in cfg["backends"].items():
                if str(bspec["device"]) in ("gpu", "hybrid") and not hw.has_gpu:
                    continue
                for qname, qspec in cfg["quant"].items():
                    if qname not in bspec.get("supports_quant", []):
                        continue
                    for batch in cfg["axes"]["batch"]:
                        for context in cfg["axes"]["context"]:
                            ap = C.analytic_prefill(m, bspec, qspec, int(batch), prompt)
                            lp = C.layerwise_prefill(m, bspec, qspec, int(batch), prompt)
                            ad = C.analytic_decode(m, bspec, qspec, int(batch), int(context))
                            ld = C.layerwise_decode(m, bspec, qspec, int(batch), int(context))
                            for a, l in ((ap, lp), (ad, ld)):
                                n_f += 1
                                r = rel(a.flops, l.flops)
                                max_f = max(max_f, r)
                                if r <= 1e-12:
                                    n_f_ok += 1
                                n_b += 1
                                r2 = rel(a.bytes_, l.bytes_)
                                max_b = max(max_b, r2)
                                if r2 <= 1e-12:
                                    n_b_ok += 1

    # ---- 排队：Erlang-C 解析 vs 离散事件模拟 ---------------------------
    n_q = n_q_ok = 0
    max_q = 0.0
    q_rows: list[dict[str, Any]] = []
    n_req = int(cfg["queue"]["n_requests"])
    warm = float(cfg["queue"]["warmup_frac"])
    tol = float(cfg["acceptance"]["max_queue_sim_abs_err"])
    seed = int(cfg["seed"])
    for s in scaling["series"]:
        for pt in s["curves"]["arrival"]:
            if not pt["feasible"] or pt["effective_batch"] is None:
                continue
            # 复算一遍读数，拿到 mu 与 c
            m = next(x for x in models if x.name == s["model"])
            hw = next(x for x in hardware if x.name == s["hardware"])
            bspec = cfg["backends"][s["backend"]]
            qspec = cfg["quant"][s["quant"]]
            cell = ENG.evaluate(m, s["backend"], bspec, s["quant"], qspec, hw,
                                int(SC.PIN["batch"]), int(SC.PIN["context"]),
                                float(pt["x"]), prompt, output,
                                max_batch_wait_s=float(
                                    cfg["queue"]["max_batch_wait_s"]))
            if not cell["feasible"]:
                continue
            r = cell["readings"]
            # ⚠️ 用 **`servers`**（配置出来的服务台数）不是 `effective_batch`
            #    （实际在跑的请求数）：排队模型的 c 是**产能**，
            #    队伍里的人不是服务台。用错的话解析与模拟会各自用不同的 c，
            #    互校「通过」了也只是因为两边都用了同一个错数。
            c_servers = int(r["servers"])
            mu = 1.0 / r["service_with_wait_s"]
            ana = Q.analytic(c_servers, mu, float(pt["x"]))
            sim = Q.simulate(c_servers, mu, float(pt["x"]), n_req, seed,
                             tag=f"{s['series_id']}|{pt['x']}",
                             warmup_frac=warm)
            if not sim["stable"] or not ana["stable"]:
                continue
            n_q += 1
            # ⚠️ 比 **`P(等待 > 0)`**：Erlang-C 的解析值 vs 模拟里的等待频率。
            #    `wq_mean` 只作为读数收进来（重尾，几千个请求都收敛不了）。
            e = abs(sim["p_wait"] - ana["p_wait"])
            max_q = max(max_q, e)
            if e <= tol:
                n_q_ok += 1
            if len(q_rows) < 24:
                q_rows.append({"series": s["series_id"], "arrival": pt["x"],
                               "c": c_servers, "rho": ana["rho"],
                               "analytic_p_wait": ana["p_wait"],
                               "sim_p_wait": sim["p_wait"], "abs_err": e,
                               "analytic_wq_mean": ana["wq_mean"],
                               "sim_wq_mean": sim["wq_mean"],
                               "mean_rel_err": rel(sim["wq_mean"], ana["wq_mean"])})

    return {
        "flops": {"n_checked": n_f, "n_agree": n_f_ok, "max_rel_err": max_f},
        "bytes": {"n_checked": n_b, "n_agree": n_b_ok, "max_rel_err": max_b},
        "queue": {"n_checked": n_q, "n_agree": n_q_ok, "max_abs_err": max_q,
                  "tol": tol, "n_requests": n_req, "warmup_frac": warm,
                  "rows": q_rows},
        "all_agree": (n_f == n_f_ok and n_b == n_b_ok and n_q == n_q_ok),
    }


# --------------------------------------------------------------------- 主流程
def run_pipeline(cfg: dict[str, Any], out_dir: str | Path | None = None,
                 write: bool = True, root: Path | None = None) -> Run:
    root = root or _PKG_ROOT
    out = Path(out_dir) if out_dir else (root / "out")
    t = time.perf_counter()
    timings: dict[str, float] = {}

    models = load_models(cfg, root)
    hardware = load_hardware(cfg, root)
    t = _tick(t, timings, "load_specs")

    products: dict[str, Any] = {
        "models.json": {"n": len(models),
                        "models": [m.to_dict() for m in models]},
        "hardware.json": {"n": len(hardware),
                          "hardware": [h.to_dict() for h in hardware]},
        "backends.json": {"n": len(cfg["backends"]),
                          "backends": cfg["backends"]},
    }

    products["inspect.json"] = step_inspect(models, hardware, cfg)
    t = _tick(t, timings, "inspect")
    products["selection.json"] = step_selection(models, hardware, cfg)
    t = _tick(t, timings, "selection")
    products["convert.json"] = step_convert(models, hardware, cfg)
    t = _tick(t, timings, "convert")
    products["warmup.json"] = step_warmup(models, hardware, cfg)
    t = _tick(t, timings, "warmup")
    products["matrix.json"] = step_matrix(models, hardware, cfg)
    t = _tick(t, timings, "matrix")
    products["stress.json"] = step_stress(products["matrix.json"], cfg)
    t = _tick(t, timings, "stress")
    products["scaling.json"] = SC.build_scaling(models, hardware, cfg)
    t = _tick(t, timings, "scaling")
    products["compare.json"] = CMP.compare_all(products["matrix.json"],
                                               models, hardware, cfg)
    t = _tick(t, timings, "compare")
    products["artifact.json"] = CMP.package_artifacts(products["compare.json"], cfg)
    t = _tick(t, timings, "package")
    products["deploy.json"] = CMP.deploy_check(products["artifact.json"], cfg)
    t = _tick(t, timings, "deploy")
    products["crosscheck.json"] = build_crosscheck(models, hardware,
                                                   products["scaling.json"], cfg)
    t = _tick(t, timings, "crosscheck")

    # ---- 门槛 / 期望 ----
    products["acceptance.json"] = INV.run_acceptance(products, cfg)
    t = _tick(t, timings, "acceptance")
    products["expectations.json"] = EXP.evaluate(products, cfg)
    t = _tick(t, timings, "expectations")

    # ---- 对抗用例（**两趟**：有一条要渲染报告，而报告要印对抗结论）----
    adv1 = INV.run_adversarial(products, cfg, models, hardware,
                               exclude=INV.REPORT_DEPENDENT_CASES)
    t = _tick(t, timings, "adversarial")

    products["config.json"] = {"resolved": cfg,
                               "protocol_digest": protocol_digest(cfg)}
    products["version.json"] = {"version": "v1-pending",
                                "protocol_digest": protocol_digest(cfg),
                                "versioned_files": list(E.VERSIONED_FILES)}
    products["summary.json"] = _summary(products, cfg)
    products["timings.json"] = {"timings": timings}

    # ⚠️ 第二趟之前先放一份 `adversarial.json`（只含第一趟的结果）：
    #    第二趟里有一条用例要**渲染报告**，而报告要读 `adversarial.json`。
    #    不放的话它在第二趟里也拿不到 ⇒ 恒红，而恒红会被误读成「渲染器坏了」。
    #    放的是第一趟的结果，第二趟跑完再用合并结果覆盖。
    products["adversarial.json"] = INV.merge_adversarial(adv1)
    adv2 = INV.run_adversarial(products, cfg, models, hardware,
                               only=INV.REPORT_DEPENDENT_CASES)
    products["adversarial.json"] = INV.merge_adversarial(adv1, adv2)
    products["summary.json"] = _summary(products, cfg)
    t = _tick(t, timings, "adversarial_report")

    digest = E.digest_of(products)
    products["version.json"] = {
        "version": f"v1-{digest}",
        "protocol_digest": protocol_digest(cfg),
        "versioned_files": list(E.VERSIONED_FILES),
        "n_versioned": len(E.VERSIONED_FILES),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    products["timings.json"] = {"timings": timings,
                                "total_wall_s": sum(timings.values())}

    # ⚠️ 正常相对导入即可：`report` 只依赖 `cost`，与 `pipeline` 没有环。
    #    `__import__(..., fromlist=...)` 的写法每次 run 都重新解析一次模块名，
    #    还会让静态检查看不到这条依赖。
    md, html = R.render(products)
    products["report.md"] = md
    products["report.html"] = html

    if write:
        out.mkdir(parents=True, exist_ok=True)
        for name, obj in products.items():
            if name.endswith(".json"):
                E.write_product(out, name, obj)
            else:
                (out / name).write_text(obj, encoding="utf-8", newline="\n")
    return Run(products=products, out_dir=out, timings=timings)


def _summary(products: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    m = products["matrix.json"]
    c = products["compare.json"]
    cc = products["crosscheck.json"]
    return {
        "n_models": m["n_models"],
        "n_hardware": m["n_hardware"],
        "n_configs": m["n_configs"],
        "n_cells": m["n_cells"],
        "matrix_expected_total": m["expected_total"],
        "matrix_gap": m["matrix_gap"],
        "n_feasible": m["n_feasible"],
        "n_infeasible_reasons": len(m["infeasible_reasons"]),
        "infeasible_reasons": m["infeasible_reasons"],
        "n_deployments": products["inspect.json"]["n_deployments"],
        "n_stages": 10,
        "crosscheck": {
            "n_checked": cc["flops"]["n_checked"] + cc["bytes"]["n_checked"]
                         + cc["queue"]["n_checked"],
            "n_agree": cc["flops"]["n_agree"] + cc["bytes"]["n_agree"]
                       + cc["queue"]["n_agree"],
            "flops": cc["flops"], "bytes": cc["bytes"],
            "queue": {k: v for k, v in cc["queue"].items() if k != "rows"},
        },
        "acceptance": {"n": products["acceptance.json"]["n"],
                       "n_passed": products["acceptance.json"]["n_passed"],
                       "n_failed": products["acceptance.json"]["n_failed"]},
        "adversarial": {"n_cases": products["adversarial.json"]["n_cases"],
                        "n_held": products["adversarial.json"]["n_held"],
                        "n_violated": products["adversarial.json"]["n_violated"]}
        if "adversarial.json" in products else {"n_cases": 0, "n_held": 0,
                                                "n_violated": 0},
        "expectations": {"n": products["expectations.json"]["n"],
                         "n_held": products["expectations.json"]["n_held"],
                         "n_refuted": products["expectations.json"]["n_refuted"],
                         "n_unmeasurable": products["expectations.json"]["n_unmeasurable"]},
        "n_backend_wins": c["n_backend_wins"],
        "n_slo_exclusions": c["n_slo_exclusions"],
        "n_pareto_total": sum(p.get("n_pareto", 0) for p in c["per_deployment"]),
        "n_decisions": sum(1 for p in c["per_deployment"] if p["decision"]),
        "scaling": {"n_series": products["scaling.json"]["n_series"],
                    "n_points": products["scaling.json"]["n_points"]},
    }
