"""显存 / 内存模型：**可行域是算出来的，不是外部给的阶梯**。

⚠️ 这是本项目与同族（`L5-quantfactory`）最不一样的地方：那边的约束
（`size_max`）是**外部阶梯**，一级一级放松；这里的约束是**内生的** ——
VRAM 占用是配置 `(batch, context)` 的函数，于是「哪些配置可行」本身就是
一个要算出来的产物（一条**双曲线** `B·C ≤ K`）。

⚠️ 两个 `util` 不是一回事，别合成一个字段：
- `kv_util`：**KV 的空间利用率**（分页 ⇒ 块内碎片小；连续预留 ⇒ 按 max_len 预留）
- `gpu_eff` / `cpu_eff`：**算力**利用率（在 `cost.py` 里）
"""

from __future__ import annotations

from typing import Any

from .cost import ACT_BYTES
from .specs import HardwareSpec, ModelSpec


def kv_alloc_tokens(backend: dict[str, Any], batch: int, context: int,
                    prompt: int, output: int) -> int:
    """KV cache **要占多少个 token 的槽位**。

    - 分页后端（`continuous`）：只按**实际用掉**的 token 分配 ⇒ `B·(prompt+output)`
    - 连续预留后端：按 `max_len` 预留 ⇒ `B·context`

    ⚠️ 这个差别是 vLLM 的立身之本，也是「为什么同样 8G 它能开更长上下文」
    的全部答案。少了它，「KV 占用」会退化成一个与后端无关的数，
    而**与后端无关的数解释不了后端之间的差别**。
    """
    used = prompt + output
    if backend.get("continuous", False):
        return batch * min(used, context)
    return batch * context


def memory_of(model: ModelSpec, backend: dict[str, Any], quant: dict[str, Any],
              hw: HardwareSpec, batch: int, context: int,
              prompt: int, output: int) -> dict[str, Any]:
    """一步推理的常驻占用。

    返回 `weight_bytes / kv_bytes / act_bytes / framework_bytes /
    gpu_bytes / ram_bytes / peak_bytes / budget_bytes / peak_frac / fits`
    """
    w_bits = int(quant["weight_bits"])
    kv_bits = int(quant["kv_bits"])

    weight = model.params * w_bits / 8.0
    kv_tok = model.kv_bytes_per_token(kv_bits)
    kv = kv_alloc_tokens(backend, batch, context, prompt, output) * kv_tok
    kv /= float(backend["kv_util"])

    # ---- 运行时激活：只对 prefill 的峰值有意义（decode 只有 1 个 token）
    act = 3.0 * batch * prompt * model.hidden * ACT_BYTES
    if not backend["flash"]:
        # 非 flash：注意力分数矩阵 `B·h·T·T` 要真的落显存
        act += batch * model.n_heads * prompt * prompt * ACT_BYTES
    act += batch * model.vocab * ACT_BYTES            # logits
    framework = float(backend["framework_bytes"])

    kind = str(backend["device"])
    if kind == "gpu":
        gpu = weight + kv + act + framework
        ram = 0.0
        budget = hw.gpu_usable_bytes
    elif kind == "cpu":
        gpu = 0.0
        ram = weight + kv + act + framework
        budget = hw.ram_bytes
    elif kind == "hybrid":
        off = float(backend["offload"])
        gpu = off * (weight + kv) + act + framework
        ram = (1.0 - off) * (weight + kv) + 2.0 * act
        budget = hw.gpu_usable_bytes
    else:
        raise ValueError(f"未知 device：{kind}")

    peak = max(gpu, ram)
    return {
        "weight_bytes": weight,
        "kv_bytes": kv,
        "kv_tokens": kv_alloc_tokens(backend, batch, context, prompt, output),
        "act_bytes": act,
        "framework_bytes": framework,
        "gpu_bytes": gpu,
        "ram_bytes": ram,
        "peak_bytes": peak,
        "budget_bytes": budget,
        "peak_frac": peak / budget if budget > 0 else float("inf"),
    }


def feasibility(model: ModelSpec, backend: dict[str, Any], quant: dict[str, Any],
                hw: HardwareSpec, batch: int, context: int,
                prompt: int, output: int, slots: int) -> tuple[bool, str]:
    """可行性判定：**原因要具体**，不能只给一个 `False`。

    ⚠️ 「不可行」如果只有一个原因，那「可行性」就不是一个维度 ——
    它只是「某个数太大了」。门槛 `min_infeasible_reasons` 就是守这件事的。
    """
    kind = str(backend["device"])
    if kind in ("gpu", "hybrid") and not hw.has_gpu:
        return False, "no_gpu"
    if prompt + output > context:
        return False, "context_too_short"
    if slots <= 0:
        return False, "no_slots"

    m = memory_of(model, backend, quant, hw, batch, context, prompt, output)
    if kind == "gpu":
        if m["gpu_bytes"] > hw.gpu_usable_bytes:
            return False, "gpu_oom"
    elif kind == "cpu":
        if m["ram_bytes"] > hw.ram_bytes:
            return False, "ram_oom"
    elif kind == "hybrid":
        if m["gpu_bytes"] > hw.gpu_usable_bytes:
            return False, "gpu_oom"
        if m["ram_bytes"] > hw.ram_bytes:
            return False, "ram_oom"
    return True, "ok"


def kv_budget_bytes(model: ModelSpec, backend: dict[str, Any],
                    quant: dict[str, Any], hw: HardwareSpec,
                    batch: int, context: int, prompt: int,
                    output: int) -> float:
    """**KV 能用的预算** = 设备容量 − 权重 − 激活 − 框架。

    ⚠️ 这个数与 `batch`、`context` **无关**（激活项只跟 prompt 走）⇒
    可行域就是 `kv_tokens · kv_per_token / kv_util ≤ 预算`，
    而 `kv_tokens = B·C`（或 `B·(prompt+output)`）⇒ **可行域是一条双曲线**。

    ⚠️ 预算 ≤ 0 意味着**权重本身就放不下** —— 这是「7B fp16 上不了 8G」
    那条结论的机制，不是一句硬编码的判断。
    """
    w_bits = int(quant["weight_bits"])
    weight = model.params * w_bits / 8.0
    act = 3.0 * batch * prompt * model.hidden * ACT_BYTES
    if not backend["flash"]:
        act += batch * model.n_heads * prompt * prompt * ACT_BYTES
    act += batch * model.vocab * ACT_BYTES
    framework = float(backend["framework_bytes"])

    kind = str(backend["device"])
    if kind == "gpu":
        cap = hw.gpu_usable_bytes
        fixed = weight + act + framework
    elif kind == "cpu":
        cap = hw.ram_bytes
        fixed = weight + act + framework
    elif kind == "hybrid":
        cap = hw.gpu_usable_bytes
        fixed = float(backend["offload"]) * weight + act + framework
    else:
        raise ValueError(f"未知 device：{kind}")
    return cap - fixed
