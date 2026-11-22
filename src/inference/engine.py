"""单格评估：**一次配置的完整 InferenceProfile**。

⚠️ 这里有一个**定点**：`batch → TPOT → 服务时间 → 系统内请求数 → 实际 batch`。
「配了 batch=64」不等于「实际同时在跑 64 个」—— 到达率低的时候批根本填不满，
于是**实际吞吐 = min(配置, 负载)**。跳过这个定点，会得到「batch 越大越好」
这种只有在上限处才成立的结论。

⚠️ 吞吐用的是**产能口径**（`eb / tpot`），不是「交付吞吐」：
在稳定区里交付吞吐恒等于 `λ·output_tokens`（稳态下到达的都被服务了），
拿来当目标**没有区分度**。这个观察本身是一条产出（见 README 附录 B）。
"""

from __future__ import annotations

from typing import Any

from . import cost as C
from . import memory as M
from . import queueing as Q
from .specs import HardwareSpec, ModelSpec

#: 定点迭代次数与阻尼（**确定**：固定次数 + 固定系数）
_FP_ITERS = 8
_FP_DAMP = 0.5


def slots_of(backend: dict[str, Any], batch: int) -> int:
    """并发服务台数上限。

    - `slots_mode = "batch"`：静态批 ⇒ 一批多少就是多少
    - `slots_mode = "fixed"`：调度器给的槽位数（continuous batching 的 256 / 单流的 1）
    """
    mode = backend.get("slots_mode", "batch")
    if mode == "fixed":
        return max(1, int(backend.get("slots_fixed", 1)))
    return max(1, int(batch))


def _prefill_decode(model: ModelSpec, backend: dict[str, Any],
                    quant: dict[str, Any], hw: HardwareSpec,
                    batch: float, context: int, prompt: int,
                    work_fn: Any) -> tuple[C.Timing, C.Timing, C.Work, C.Work]:
    wp = work_fn["prefill"](model, backend, quant, int(round(batch)), prompt)
    wd = work_fn["decode"](model, backend, quant, int(round(batch)), context)
    tp = C.seconds_of(wp, backend, quant, hw, model.n_layers,
                      tokens_moved=int(round(batch)) * prompt)
    td = C.seconds_of(wd, backend, quant, hw, model.n_layers,
                      tokens_moved=int(round(batch)))
    return tp, td, wp, wd


def evaluate(model: ModelSpec, backend_name: str, backend: dict[str, Any],
             quant_name: str, quant: dict[str, Any], hw: HardwareSpec,
             batch: int, context: int, arrival: float,
             prompt: int, output: int,
             work_fn: Any | None = None,
             max_batch_wait_s: float = 0.05) -> dict[str, Any]:
    """评估**一格**，返回完整读数（不可行时也返回，带 `reason`）。

    返回的 `readings` 是「**模型预测**」—— 本机没有 GPU，
    「这个配置有多快」是 roofline + 排队论算出来的，不是测出来的。
    """
    wf = work_fn or {"prefill": C.analytic_prefill, "decode": C.analytic_decode}
    base: dict[str, Any] = {
        "model": model.name,
        "hardware": hw.name,
        "backend": backend_name,
        "quant": quant_name,
        "batch": batch,
        "context": context,
        "arrival_rate": arrival,
    }

    # ---- 可行性（设备 / 量化支持 / 显存）--------------------------------
    # ⚠️ `mem` **先算**：不可行的格子也要带占用明细。
    #    少了它，「为什么不可行」只能靠 `reason` 一个字符串回答，
    #    而「差多少字节」这个问题在产物里就没有答案了。
    slots = slots_of(backend, batch)
    mem = M.memory_of(model, backend, quant, hw, batch, context, prompt, output)
    if quant_name not in backend.get("supports_quant", []):
        return {**base, "feasible": False, "reason": "quant_unsupported",
                "readings": {}, "slots": slots, "memory": mem}
    ok, reason = M.feasibility(model, backend, quant, hw, batch, context,
                               prompt, output, slots)

    if not ok:
        return {**base, "feasible": False, "reason": reason,
                "slots": slots, "memory": mem, "readings": {}}

    # ---- 定点：实际 batch ----------------------------------------------
    # ⚠️ **服务台数 c 与实际 batch eb 是两个数**，别合成一个：
    #    - `c = min(batch, slots)`：这台机器**能**同时处理多少请求（配置的产能）
    #    - `eb = min(c, λ·S)`：实际同时在跑多少（Little 定律作用在**服务**上，
    #      不含排队等待 —— 用 `λ·W` 会把「队伍里的人」也算进批里）
    #    合成一个的后果：低负载时 c 塌到 1 ⇒ M/M/1 ⇒ ρ 不小 ⇒ **P99 反而爆炸**，
    #    「负载越低延迟越高」这个荒谬的读数就是这么来的。
    cap = min(batch, slots)
    c_servers = max(1, cap)
    eb = float(c_servers)
    q: dict[str, Any] = {"stable": True, "rho": 0.0, "wq_mean": 0.0, "wq_p99": 0.0}
    # ⚠️ 静态批的凑批等待要**封顶**（真实服务器用 batching timeout，
    #    不会真的等到凑满）。不封顶的话 λ=0.25、B=16 会算出 30 秒的等待 ——
    #    数字是对的，但描述的事（无限期等待）不是真实系统做的事。
    max_wait = float(max_batch_wait_s)

    def _service_at(b: int) -> tuple[float, float, float, Any, Any, Any]:
        tp_, td_, wp_, wd_ = _prefill_decode(model, backend, quant, hw,
                                             b, context, prompt, wf)
        pre = tp_.seconds
        tpot = td_.seconds
        s = pre + output * tpot
        w = 0.0 if backend.get("continuous", False) else min(
            Q.static_batching_wait(b, arrival), max_wait)
        return pre, tpot, s + w, tp_, td_, (wp_, wd_)

    for _ in range(_FP_ITERS):
        _, _, S_eff, tp, td, wpwd = _service_at(max(1, int(round(eb))))
        mu = 1.0 / S_eff if S_eff > 0 else float("inf")
        q = Q.analytic(c_servers, mu, arrival)
        if not q["stable"]:
            break
        busy = arrival * S_eff          # **服务**部分的 Little，不含等待
        target = min(float(c_servers), max(1.0, busy))
        eb = _FP_DAMP * eb + (1.0 - _FP_DAMP) * target

    if not q["stable"]:
        return {**base, "feasible": False, "reason": "unstable",
                "slots": slots, "effective_batch": cap, "memory": mem,
                "rho": q["rho"], "readings": {}}

    eb_int = max(1, int(round(eb)))
    # ⚠️ 用**最终** eb 重算一次 —— 读数是收敛后的值，不是迭代中途的
    prefill_s, tpot_s, S_eff, tp, td, (wp, wd) = _service_at(eb_int)
    S = prefill_s + output * tpot_s          # 纯服务时间（不含凑批等待）
    wait_batch = S_eff - S
    mu = 1.0 / S_eff if S_eff > 0 else float("inf")
    q = Q.analytic(c_servers, mu, arrival)
    if not q["stable"]:
        return {**base, "feasible": False, "reason": "unstable",
                "slots": slots, "effective_batch": eb_int, "memory": mem,
                "rho": q["rho"], "readings": {}}

    # ---- 吞吐：两个口径，**不能合成一个** ------------------------------
    # ⚠️ `eb / tpot` 是**只算 decode 阶段**的速率：它假设每个请求在整个
    #    `O·tpot` 里都在出 token，却忘了 **prefill 也占着同一块 GPU**。
    #    实测这个数与「交付吞吐」自相矛盾（59 个请求 × 128 token / 7.4 s
    #    = 1024 tok/s，而 `eb/tpot` 报 2500 tok/s）。
    #    ⇒ 报**满批产能** `c·O / S(c)`：一批 c 个请求走完一个完整周期
    #      （prefill + O 步 decode）能吐多少 token。这是 benchmark 的口径。
    prefill_full, tpot_full, S_full, tp_full, td_full, _wf = _service_at(c_servers)
    capacity_tokens_per_s = (c_servers * output / S_full) if S_full > 0 else 0.0
    capacity_req_s = c_servers / S_eff if S_eff > 0 else 0.0
    served_req_s = min(arrival, capacity_req_s)
    # 交付吞吐：稳态下到达的都被服务 ⇒ `min(λ, 产能)·O`。
    # ⚠️ 它在稳定区**恒等于 λ·O**，拿来当目标没有区分度 —— 这正是要报
    #    「产能」而不是「交付」的原因（这条观察本身是产出，见 README）。
    delivered_tokens_per_s = served_req_s * output

    readings = {
        "effective_batch": eb_int,
        "slots": slots,
        # ⚠️ `servers` 是排队模型的 **c**（配置出来的产能），
        #    `effective_batch` 是实际在跑的请求数。两个都要留 ——
        #    合成一个会让「队伍里的人」被当成服务台（见 `pipeline.build_crosscheck`）。
        "servers": c_servers,
        "rho": q["rho"],
        # ---- 三个延迟读数 ----
        "ttft_service_ms": prefill_s * 1e3,
        "ttft_ms": (q["wq_mean"] + prefill_s) * 1e3,
        "tpot_ms": tpot_s * 1e3,
        "p99_ms": (q["wq_p99"] + S_eff) * 1e3,
        # ---- 两个吞吐读数（**产能**与**交付**）----
        "tokens_per_s": capacity_tokens_per_s,
        "delivered_tokens_per_s": delivered_tokens_per_s,
        "capacity_req_per_s": capacity_req_s,
        "served_req_per_s": served_req_s,
        "prefill_full_s": prefill_full,
        "decode_full_s": tpot_full,
        "service_full_s": S_full,
        "bound_prefill_full": tp_full.bound,
        "bound_decode_full": td_full.bound,
        # ---- 占用 ----
        "peak_bytes": mem["peak_bytes"],
        "peak_frac": mem["peak_frac"],
        "gpu_bytes": mem["gpu_bytes"],
        "ram_bytes": mem["ram_bytes"],
        # ---- roofline ----
        "prefill_s": prefill_s,
        "decode_s": tpot_s,
        "bound_prefill": tp.bound,
        "bound_decode": td.bound,
        "prefill_flops": wp.flops,
        "prefill_bytes": wp.bytes_,
        "decode_flops": wd.flops,
        "decode_bytes": wd.bytes_,
        "arithmetic_intensity_decode": C.arithmetic_intensity(wd.flops, wd.bytes_),
        "wait_batch_s": wait_batch,
        "wq_mean_s": q["wq_mean"],
        "wq_p99_s": q["wq_p99"],
        "service_s": S,
        "service_with_wait_s": S_eff,
    }
    return {**base, "feasible": True, "reason": "ok",
            "slots": slots, "effective_batch": eb_int,
            "memory": mem, "readings": readings,
            "timing": {"prefill": tp.as_dict(), "decode": td.as_dict()}}
