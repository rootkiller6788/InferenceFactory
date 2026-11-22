"""配置：**口径与门槛都在配置里，代码里不硬编码阈值**。

⚠️ 改了 `DEFAULTS` 之后必须跑 `python main.py dump-config` 重新导出
`configs/default.json` —— `deep_merge` 里**列表整体替换、磁盘优先**，
磁盘上的旧列表会**静默盖掉**新声明。保险丝 `no_stale_config_keys`
只拦「磁盘上有而代码里没有的键」，拦不住「两边都有但值不同」。
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """配置错误（缺键 / 类型不对 / 有代码里不存在的键）。"""


DEFAULTS: dict[str, Any] = {
    # ------------------------------------------------------------------ 随机
    "seed": 20261003,

    # ------------------------------------------------------------------ 数据
    "data": {
        # ⚠️ 相对路径锚**项目根**（`cli._anchor_args` 只锚 CLI 参数，
        #    这两个是数据目录 ⇒ 由 `specs.load_*` 自己锚 `_PKG_ROOT`）。
        "models_dir": "data/models",
        "hardware_dir": "data/hardware",
    },

    # ------------------------------------------------------------------ 负载
    # ⚠️ 「请求长什么样」不是设计轴 —— 它是**固定的测试负载**。
    #    把它放进搜索空间会让「backend 好不好」混进「负载是什么」。
    "workload": {
        "prompt_tokens": 512,
        "output_tokens": 128,
    },

    # ------------------------------------------------------------------ 设计轴
    # 笛卡尔积 = 4 · 3 · 3 · 3 · 3 = **324 条配置**；
    # × 3 个模型 × 2 台硬件 = **1944 格**（全枚举，缺口必须为 0）。
    "axes": {
        "backend": ["pytorch-eager", "vllm-paged", "llamacpp-cpu", "llamacpp-hybrid"],
        "quant": ["fp16", "int8", "int4"],
        "batch": [1, 8, 64],
        "context": [2048, 16384, 131072],
        # ⚠️ 三档要落在三个**不同区间**：0.5 = 谁都扛得住，4 = 只有 GPU 扛得住，
        #    32 = 只有最好的那几个配置扛得住（其余 ρ ≥ 1 ⇒ unstable）。
        #    三档都一样的话，「到达率是一根轴」就是假的。
        # ⚠️ 这根轴必须**跨在产能两侧**，否则它不是一根轴而是一个常数开关。
        # 第一版取 [0.5, 2.0, 8.0]：λ=8 那一段 648 格里只有 3 格可行
        # （`unstable` 626 条）⇒ 高位整段是死的，什么都测不出来。
        # 实测可行域的产能中位数 5.2 req/s ⇒ 下移一档到 [0.25, 1.0, 4.0]：
        # λ=1 在产能之内，λ=4 在产能边缘之外，两端都有东西。
        "arrival": [0.25, 1.0, 4.0],   # req/s（**到达率**，不是并发数）
    },

    # ------------------------------------------------------------------ 后端目录
    # ⚠️ `gpu_eff` / `cpu_eff` 是**算力利用率**；`kv_util` 是 **KV 显存的
    #    空间利用率**（非分页后端要按 max_len 预留连续显存 ⇒ 利用率低）。
    #    这两个 `util` 说的不是同一件事，别合成一个字段。
    # ⚠️ `slots_mode` 决定**排队模型的服务台数 c**，它是这个后端区别于别的后端
    # 最重要的一根轴 —— 比 `gpu_eff` 重要得多：单流服务（c=1）的 P99 由排队主导，
    # 与「它算得多快」几乎无关。
    #   - `"fixed"`：c = `slots_fixed`（**与 batch 无关**）。vllm 的 256 槽是
    #     调度器给的；llamacpp 若设成 1 就是严格单流 ⇒ batch 轴在它上面**全死**
    #     （batch=1/8/64 读数逐位相同），而 c=1 ⇒ M/M/1 ⇒ ρ 不低 ⇒ P99 必然爆。
    #   - `"batch"`：c = 配置的 batch（静态批 / 能凑多少算多少）。
    "backends": {
        "pytorch-eager": {
            "device": "gpu",
            "continuous": False,
            "kv_util": 0.60,
            "gpu_eff": 0.55,
            "slots_mode": "batch",     # 服务台数 = batch
            "slots_fixed": 0,
            "batching": "static",      # 静态批：要凑满一批才发
            "launch_overhead_us": 18.0,
            "scheduler_overhead_ms": 0.0,
            "flash": False,            # 非 flash ⇒ attention 显存 O(P²)
            "framework_bytes": 550_000_000,
            "supports_quant": ["fp16", "int8"],
        },
        "vllm-paged": {
            "device": "gpu",
            "continuous": True,
            "kv_util": 0.95,
            "gpu_eff": 0.85,
            "slots_mode": "fixed",
            "slots_fixed": 256,        # continuous batching：调度器给的槽位数
            "batching": "continuous",
            "launch_overhead_us": 6.0,
            "scheduler_overhead_ms": 0.5,   # 调度器 + 抢占判定
            "flash": True,
            "framework_bytes": 900_000_000,
            "supports_quant": ["fp16", "int8", "int4"],
        },
        "llamacpp-cpu": {
            "device": "cpu",
            "continuous": False,
            "kv_util": 0.90,
            "cpu_eff": 0.45,
            # ⚠️ 第一版这里写的是 `fixed / 1`（「单流」），后果：
            #    c 恒为 1 ⇒ batch 轴在这条后端上**全死**（batch=1/8/64 的读数
            #    逐位相同），且 M/M/1 在 ρ=0.32 时 P99 就是 7.6 s
            #    ⇒ CPU 部署**永远**过不了 SLO，跟 CPU 有多快毫无关系。
            "slots_mode": "batch",     # llama.cpp server 按批跑，不是严格单流
            "slots_fixed": 0,
            "batching": "static",
            "launch_overhead_us": 2.0,
            "scheduler_overhead_ms": 0.0,
            "flash": False,
            "kernels_per_layer": 8,
            "framework_bytes": 40_000_000,
            "supports_quant": ["fp16", "int8", "int4"],
        },
        "llamacpp-hybrid": {
            "device": "hybrid",
            "continuous": False,
            "kv_util": 0.85,
            "gpu_eff": 0.60,
            "cpu_eff": 0.45,
            "offload": 0.60,           # 60% 的层放 GPU，其余在 CPU
            "slots_mode": "batch",
            "slots_fixed": 0,
            "batching": "static",
            "launch_overhead_us": 10.0,
            "scheduler_overhead_ms": 0.2,
            "flash": True,
            "framework_bytes": 320_000_000,
            "supports_quant": ["int8", "int4"],
        },
    },

    # ------------------------------------------------------------------ 量化
    # ⚠️ **`compute_eff_scale` 是量化的代价**：权重量化省的是**带宽**，
    #    不是算力 —— 推理时要反量化，算力侧反而略慢。
    #    少了这一项，「量化越狠越好」会成为模型的假设，而不是模型的结论。
    "quant": {
        "fp16": {"weight_bits": 16, "kv_bits": 16, "compute_eff_scale": 1.00},
        "int8": {"weight_bits": 8, "kv_bits": 8, "compute_eff_scale": 0.96},
        "int4": {"weight_bits": 4, "kv_bits": 4, "compute_eff_scale": 0.92},
    },

    # ------------------------------------------------------------------ 排队
    "queue": {
        # 离散事件模拟（互校用）：每条 (部署单元, backend) 的 concurrency 曲线上
        # 每个点跑 `n_requests` 个请求。
        "n_requests": 4000,
        # 模拟丢弃前 `warmup_frac` 的请求（从空系统开始 ⇒ 前段等待系统性偏低）
        "warmup_frac": 0.4,
        # M/M/c 不稳定的判据：`rho >= 1 - eps`（ρ 真等于 1 时等待 → ∞）。
        "rho_epsilon": 1e-9,
        # 静态批的 batching timeout（秒）：凑不满就发车，不会无限等。
        "max_batch_wait_s": 0.05,
    },

    # ------------------------------------------------------------------ SLO
    "slo": {
        "p99_ms": 2000.0,   # P99 端到端延迟上限（含排队）
        "tpot_ms": 120.0,   # 单 token 生成时间上限
    },

    # ------------------------------------------------------------------ 门槛
    # ⚠️ 门槛是**可调的达标线**（写在配置里，改了不需要改代码）；
    #    对抗用例是**不可调的不变量**（配置里没有阈值，见 `invariants.py`）。
    "acceptance": {
        "matrix_gap_max": 0,              # 全枚举 ⇒ 缺口必须为 0
        "min_n_cells": 1000,
        "min_feasible_cells": 100,
        "min_infeasible_reasons": 3,      # 「可行性」要是真维度，不能只有一种原因
        "min_batch_speedup": 3.0,         # batch 轴上吞吐的加速比
        "min_context_tpot_growth": 1.2,   # context 拉长后 TPOT 的涨幅
        "min_p99_growth_with_load": 2.0,  # 到达率上来后 P99 的涨幅
        "max_crosscheck_rel_err": 1e-12,  # 双实现互校（解析 vs 逐层）
        # 排队互校的界：比的是 **`P(等待>0)`**（伯努利，方差小），
        # 不是平均等待（重尾，几千个请求都收敛不了）。绝对误差界。
        "max_queue_sim_abs_err": 0.05,
        # ⚠️ 不是「可行域 == 双曲线」（那是恒等式，验了等于没验），
        #    而是「长上下文下 KV 是不是真的占了显存的大头」——
        #    这条**会**失败：若激活项或框架开销被高估，KV 就不是主角。
        "min_kv_share_peak": 0.5,
        "min_roofline_bound_variety": 2,  # 矩阵里 compute-bound 与 memory-bound 都要有
        "min_backend_wins": 2,            # 至少两个 backend 赢过（否则「选 backend」是假的）
        "min_models_with_oom": 1,         # 至少一个模型在 8G 上放不下
        "min_slo_exclusions": 1,          # 至少有一个「吞吐最优」点被 SLO 淘汰
        "min_pareto_points": 5,           # 前沿至少 5 个点
    },

    # ------------------------------------------------------------------ 目标
    # ------------------------------------------------------------------ 目标
    # ⚠️ `footprint` 用 **`peak_frac`**（占用 / 该设备的预算）而不是 `peak_bytes`：
    #    GPU 后端比的是显存、CPU 后端比的是内存，**绝对字节数不是同一个量**。
    #    用字节数会让「8G 上用了 7G」和「64G 上用了 7G」看起来一样。
    "objectives": {
        "throughput": {"key": "tokens_per_s", "direction": 1},
        "latency": {"key": "p99_ms", "direction": -1},
        "footprint": {"key": "peak_frac", "direction": -1},
    },
}

#: 接受布尔/数值/字符串的配置键（递归校验用）
_KEY_TYPES: dict[str, type] = {
    "seed": int,
    "matrix_gap_max": int,
}


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """递归合并，**列表与标量整体替换**（不是 extend）。

    ⚠️ 列表整体替换是**故意**的：配置里的列表是「枚举出来的轴」，
    半合并会得到「代码里的轴 + 磁盘上的轴」这种谁都没声明过的并集。
    """
    out = deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def no_stale_config_keys(cfg: dict[str, Any],
                         defaults: dict[str, Any] | None = None,
                         path: str = "") -> list[str]:
    """磁盘配置里有、但代码里已经没有的键。

    ⚠️ 这条保险丝**只拦这一侧**。另一侧（代码里新增、磁盘上还没有）不需要拦，
    因为 `deep_merge` 会用默认值补上；而「磁盘上有、代码里没了」意味着
    **你以为在改的那个键其实已经没人读了** —— 症状是「改了配置没反应」。
    """
    d = DEFAULTS if defaults is None else defaults
    bad: list[str] = []
    for k, v in cfg.items():
        here = f"{path}.{k}" if path else k
        if k not in d:
            bad.append(here)
        elif isinstance(v, dict) and isinstance(d[k], dict):
            bad.extend(no_stale_config_keys(v, d[k], here))
    return bad


def validate_config(cfg: dict[str, Any]) -> None:
    """结构性校验（缺键 / 类型 / 轴非空 / 后端引用完整）。"""
    for group in ("data", "workload", "axes", "backends", "quant", "queue",
                  "slo", "acceptance", "objectives"):
        if group not in cfg:
            raise ConfigError(f"配置缺少 `{group}` 段")
    for axis, values in cfg["axes"].items():
        if not isinstance(values, list) or not values:
            raise ConfigError(f"设计轴 `{axis}` 必须是非空列表")
    for b in cfg["axes"]["backend"]:
        if b not in cfg["backends"]:
            raise ConfigError(f"设计轴声明了后端 `{b}`，但 `backends` 里没有它")
    for q in cfg["axes"]["quant"]:
        if q not in cfg["quant"]:
            raise ConfigError(f"设计轴声明了量化 `{q}`，但 `quant` 里没有它")
    if cfg["workload"]["prompt_tokens"] <= 0 or cfg["workload"]["output_tokens"] <= 0:
        raise ConfigError("`workload` 的 token 数必须为正")


def protocol_digest(cfg: dict[str, Any]) -> str:
    """协议指纹：口径（轴 / 后端 / 量化 / 门槛 / 目标）的函数。"""
    blob = json.dumps(
        {k: cfg[k] for k in ("axes", "backends", "quant", "workload",
                             "acceptance", "objectives", "slo", "queue")},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path | None = None,
                root: Path | None = None) -> dict[str, Any]:
    """读配置（缺省时用 `DEFAULTS`）。

    ⚠️ 缺文件时**返回 `DEFAULTS` 的副本**而不是报错 —— 这是知情的取舍：
    「跑不起来」比「静默换了一份口径」好查得多，所以这里**不让静默回落**，
    调用方要么拿到磁盘配置，要么拿到同一份 `DEFAULTS`，两者内容一致时
    「今天恰好对」与「对」无法区分 ⇒ 由 `dump-config` 负责让它们永远一致。
    """
    if path is None:
        cfg = deepcopy(DEFAULTS)
    else:
        p = Path(path)
        if not p.is_absolute() and root is not None:
            p = Path(root) / p
        if not p.exists():
            raise ConfigError(f"配置文件不存在：{p}")
        disk = json.loads(p.read_text(encoding="utf-8"))
        stale = no_stale_config_keys(disk)
        if stale:
            raise ConfigError(
                "配置里有代码中不存在的键：" + "、".join(stale)
                + "（改完 `config.py` 的 `DEFAULTS` 后要跑 "
                "`python main.py dump-config` 重新导出）"
            )
        cfg = deep_merge(DEFAULTS, disk)
    validate_config(cfg)
    return cfg


def dump_config(cfg: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                 encoding="utf-8", newline="\n")
    return p
