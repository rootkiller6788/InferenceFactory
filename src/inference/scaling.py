"""三条 scaling 曲线 —— **这个项目的核心输出**。

设计稿里那句话是对的：「单点数字（跑出 50 tokens/s）几乎无信息量」。
真正说明问题的是曲线：batch 从 1 加到 128 吞吐怎么变、context 从 512 拉到
128K 延迟怎么涨、到达率上来后 P99 怎么劣化。

⚠️ 三条曲线**各扫一根轴，其余固定在同一个点** —— 这样曲线上的每个点
只差一个变量，「曲线拐了个弯」才能归因到那根轴上。
"""

from __future__ import annotations

from typing import Any

from .engine import evaluate
from .specs import HardwareSpec, ModelSpec

#: 三条轴各自的网格（比矩阵的轴**密**，因为曲线要看拐点在哪）
BATCH_GRID: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
CONTEXT_GRID: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384, 32768,
                                 65536, 131072)
ARRIVAL_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0,
                                   64.0, 128.0)

#: 扫某根轴时，其余两根固定在**同一个**点上
PIN: dict[str, Any] = {"batch": 16, "context": 2048, "arrival": 0.5}


def _series(model: ModelSpec, hw: HardwareSpec, backend_name: str,
            backend: dict[str, Any], quant_name: str, quant: dict[str, Any],
            cfg: dict[str, Any], axis: str, grid: tuple[Any, ...],
            prompt: int, output: int) -> list[dict[str, Any]]:
    rows = []
    for v in grid:
        b, c, a = int(PIN["batch"]), int(PIN["context"]), float(PIN["arrival"])
        if axis == "batch":
            b = int(v)
        elif axis == "context":
            c = int(v)
        else:
            a = float(v)
        cell = evaluate(model, backend_name, backend, quant_name, quant, hw,
                        b, c, a, prompt, output,
                        max_batch_wait_s=float(cfg["queue"]["max_batch_wait_s"]))
        rd = cell.get("readings") or {}
        rows.append({
            "x": v,
            "feasible": bool(cell["feasible"]),
            "reason": cell["reason"],
            "tokens_per_s": rd.get("tokens_per_s"),
            "p99_ms": rd.get("p99_ms"),
            "ttft_ms": rd.get("ttft_ms"),
            "tpot_ms": rd.get("tpot_ms"),
            "peak_frac": rd.get("peak_frac"),
            "effective_batch": rd.get("effective_batch"),
            "bound_decode": rd.get("bound_decode"),
            "rho": rd.get("rho"),
        })
    return rows


def build_scaling(models: tuple[ModelSpec, ...],
                  hardware: tuple[HardwareSpec, ...],
                  cfg: dict[str, Any]) -> dict[str, Any]:
    """`(模型 × 硬件 × 后端 × 量化)` × 三条轴。"""
    prompt = int(cfg["workload"]["prompt_tokens"])
    output = int(cfg["workload"]["output_tokens"])
    axes = {
        "batch": {"x_label": "batch size", "grid": list(BATCH_GRID),
                  "pinned": {"context": PIN["context"], "arrival": PIN["arrival"]}},
        "context": {"x_label": "context (tokens)", "grid": list(CONTEXT_GRID),
                    "pinned": {"batch": PIN["batch"], "arrival": PIN["arrival"]}},
        "arrival": {"x_label": "arrival rate (req/s)", "grid": list(ARRIVAL_GRID),
                    "pinned": {"batch": PIN["batch"], "context": PIN["context"]}},
    }
    series: list[dict[str, Any]] = []
    for m in models:
        for hw in hardware:
            for bname, bspec in cfg["backends"].items():
                for qname in cfg["axes"]["quant"]:
                    if qname not in bspec.get("supports_quant", []):
                        continue
                    qspec = cfg["quant"][qname]
                    series.append({
                        "model": m.name,
                        "hardware": hw.name,
                        "backend": bname,
                        "quant": qname,
                        "series_id": f"{m.name}@{hw.name}|{bname}|{qname}",
                        "curves": {
                            ax: _series(m, hw, bname, bspec, qname, qspec, cfg,
                                        ax, tuple(meta["grid"]), prompt, output)
                            for ax, meta in axes.items()
                        },
                    })
    n_points = sum(len(s["curves"][ax]) for s in series for ax in axes)
    return {"pin": dict(PIN), "axes": axes, "n_series": len(series),
            "n_points": n_points, "series": series}
