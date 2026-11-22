"""M/M/c 排队：**解析（Erlang-C）+ 离散事件模拟**，两条独立实现互校。

⚠️ 这是本项目唯一一根**随机过程**轴。别的轴（batch / context / 量化）都是
确定性映射 —— 改一次输入只有一个输出。到达率不一样：同一个配置的
**每一次运行都是不同的**，能说的是「稳态分布」。于是这一层的读数
（P99）是**分位数**，不是一个数；而「P99 是多少」这件事本身要能被
**第二条实现**对上，否则它只是「把公式抄了一遍」。

两条实现：
1. **解析**：Erlang-C 闭式 ⇒ `W_q` 均值与 `P99`（指数尾 `C·e^{-(cμ-λ)t}`）
2. **模拟**：真的按 FIFO 排 M/M/c 队列（确定性 LCG，见 `rng`），
   丢弃前 `warmup_frac` 的请求后统计
"""

from __future__ import annotations

import heapq
import math
from typing import Any

from .rng import stream_for

#: ρ 到 1 的距离小于这个数就判**不稳定**（等待 → ∞，P99 无意义）
RHO_EPS = 1e-9


def erlang_b(c: int, a: float) -> float:
    """Erlang-B（阻塞概率）递推：`B(n,a) = a·B(n-1,a) / (n + a·B(n-1,a))`。

    ⚠️ 用**递推**而不是 `a^n/n!`：后者在 `a` 大、`c` 大时先溢出再下溢，
    而递推的每一步都落在 (0,1) 里，数值上不会炸。
    """
    b = 1.0
    for n in range(1, c + 1):
        b = a * b / (n + a * b)
    return b


def erlang_c(c: int, a: float) -> float:
    """Erlang-C（等待概率）：`C = B / (1 - ρ(1 - B))`，`ρ = a/c`。"""
    if c <= 0:
        return float("inf")
    rho = a / c
    if rho >= 1.0:
        return float("inf")
    b = erlang_b(c, a)
    denom = 1.0 - rho * (1.0 - b)
    if denom <= 0.0:
        return float("inf")
    return b / denom


def analytic(c: int, mu: float, lam: float) -> dict[str, Any]:
    """M/M/c 的解析读数。

    - `mu`：单服务台的服务率（req/s），`c` 个服务台
    - `lam`：到达率（req/s）
    - 返回 `rho / wq_mean / wq_p99 / p_wait / stable`

    ⚠️ `rho >= 1` 时返回 `stable=False` 而不是给一个巨大的数 ——
    「队列爆炸」和「队列很长」是两件事，混在一起会让下游的 SLO 判定
    去比较 `inf`，而 `inf <= 2000` 是 `False` 这件事靠运气才对。
    """
    if c <= 0 or mu <= 0:
        return {"stable": False, "reason": "no_server", "rho": float("inf"),
                "wq_mean": float("inf"), "wq_p99": float("inf"),
                "p_wait": float("inf")}
    a = lam / mu
    rho = a / c
    if rho >= 1.0 - RHO_EPS:
        return {"stable": False, "reason": "unstable", "rho": rho,
                "wq_mean": float("inf"), "wq_p99": float("inf"),
                "p_wait": 1.0}
    if lam <= 0:
        return {"stable": True, "reason": "idle", "rho": 0.0,
                "wq_mean": 0.0, "wq_p99": 0.0, "p_wait": 0.0}
    cc = erlang_c(c, a)
    slack = c * mu - lam                      # > 0
    wq_mean = cc / slack
    # P(W_q > t) = C·e^{-slack·t}  ⇒  t_99 = ln(100·C)/slack
    wq_p99 = math.log(100.0 * cc) / slack if cc > 0.01 else 0.0
    if wq_p99 < 0.0:
        wq_p99 = 0.0
    return {"stable": True, "reason": "ok", "rho": rho, "wq_mean": wq_mean,
            "wq_p99": wq_p99, "p_wait": cc}


def simulate(c: int, mu: float, lam: float, n_requests: int, seed: int,
             tag: str = "", warmup_frac: float = 0.4) -> dict[str, Any]:
    """离散事件模拟 M/M/c FIFO（**确定性**：LCG + 固定标签）。

    ⚠️ 丢弃前 `warmup_frac` 的请求：从空系统开始的系统，前若干请求的
    等待**系统性偏低**（队列还没积压起来）。不丢的话，在高 ρ 处模拟值会
    稳定地低于解析值 —— 症状是「互校在高负载处才失败」，看起来像
    「模拟不准」，实际是**忘了热机**。
    """
    if c <= 0 or mu <= 0 or lam <= 0:
        return {"stable": False, "reason": "degenerate", "n": 0,
                "wq_mean": float("inf"), "wq_p99": float("inf")}
    rng = stream_for(tag, "mmc", seed=seed)
    # 到达时刻（累积指数间隔）与服务时长
    # ⚠️ 批量采样：取数顺序与逐个 `exponential` 完全相同（先 λ 后 μ）。
    t = 0.0
    arrivals: list[float] = []
    for gap in rng.exponentials(n_requests, lam):
        t += gap
        arrivals.append(t)
    services = rng.exponentials(n_requests, mu)

    free: list[float] = [0.0] * c     # 每个服务台的空闲时刻（小根堆）
    heapq.heapify(free)
    waits: list[float] = []
    for a, s in zip(arrivals, services):
        earliest = free[0]
        start = a if a > earliest else earliest
        waits.append(start - a)
        heapq.heapreplace(free, start + s)

    cut = int(n_requests * warmup_frac)
    tail = waits[cut:]
    if not tail:
        return {"stable": False, "reason": "no_tail", "n": 0,
                "wq_mean": float("inf"), "wq_p99": float("inf")}
    tail_sorted = sorted(tail)
    k = min(len(tail_sorted) - 1, int(math.ceil(0.99 * len(tail_sorted))) - 1)
    n_wait = sum(1 for w in tail if w > 1e-12)
    return {"stable": True, "reason": "ok", "n": len(tail),
            "wq_mean": sum(tail) / len(tail),
            "wq_p99": tail_sorted[max(k, 0)],
            "wq_max": tail_sorted[-1],
            # ⚠️ **`p_wait` 才是能拿来互校的量**：它是个伯努利频率，
            #    方差 `p(1-p)/n` ⇒ n=2400 时标准差 ~0.01。
            #    而 `wq_mean` 在重尾分布下收敛极慢（ρ=0.9 时几千个请求
            #    都不够），拿它做互校的界，界就只能松到没有意义。
            "p_wait": n_wait / len(tail)}


def static_batching_wait(batch: int, lam: float) -> float:
    """静态批的**凑批等待**：一个请求平均要等 `(B-1)/(2λ)` 才凑满一批。

    ⚠️ 连续批（continuous batching）没有这一项 —— 请求随时可以进。
    少了这一项，「静态批和连续批的差别」会只剩 KV 利用率，
    而 continuous batching 真正的卖点是**不等批**。
    """
    if lam <= 0 or batch <= 1:
        return 0.0
    return (batch - 1) / (2.0 * lam)
