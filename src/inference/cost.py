"""Roofline 成本模型：**两条独立实现同一组数**，互校用。

⚠️ 为什么要有两条（这正是这个项目的第二层真值）：
`analytic_*` 是**闭式**（`flops = 2·W_mat·B·T + 4·B·h·d_h·T²·L`），
`layerwise_*` 是**逐层累加**（真的把 L 层的 qkv / attn / o / ffn 各算一遍）。
它们描述同一件事却走**完全不同的算术路径** —— 于是「闭式里漏了 o_proj」
这类错误会立刻变红，而不是静默给出一个看起来合理的数。

⚠️ 时间是 **工作量的纯函数**（`seconds_of`），两条实现共用它 ——
互校比的是**工作量**（谁对谁错一目了然），不是**时间**
（时间里混了设备选择、PCIe、launch 开销，混在一起就说不清哪错了）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .specs import HardwareSpec, ModelSpec

#: 激活值字节宽（fp16 推理）
ACT_BYTES = 2.0


class DeviceError(ValueError):
    """后端声明了一个不认识的 device（或要的设备机器上没有）。"""


@dataclass(frozen=True)
class Work:
    """一次前向的**工作量**（与设备无关）。"""

    flops: float
    bytes_: float
    weights_bytes: float
    kv_bytes: float
    act_bytes: float

    def as_dict(self) -> dict[str, float]:
        return {"flops": self.flops, "bytes": self.bytes_,
                "weights_bytes": self.weights_bytes,
                "kv_bytes": self.kv_bytes, "act_bytes": self.act_bytes}


@dataclass(frozen=True)
class Timing:
    """工作量落在某台机器上的时间。"""

    seconds: float
    bound: str            # "compute" | "memory"
    compute_s: float
    memory_s: float
    overhead_s: float
    transfer_s: float

    def as_dict(self) -> dict[str, Any]:
        return {"seconds": self.seconds, "bound": self.bound,
                "compute_s": self.compute_s, "memory_s": self.memory_s,
                "overhead_s": self.overhead_s, "transfer_s": self.transfer_s}


# --------------------------------------------------------------------- 闭式
def _w_mat(model: ModelSpec) -> int:
    """参与 matmul 的权重数：每层五个矩阵 + lm_head（与 embed 绑定 ⇒ 用 embed_size）。"""
    return model.n_layers * model.params_per_layer + model.embed_size


def analytic_prefill(model: ModelSpec, backend: dict[str, Any],
                     quant: dict[str, Any], batch: int, prompt: int) -> Work:
    """Prefill：T = prompt 个 token × B 个请求，一次前向。

    - 算力 `2·W_body·B·T`（主干 matmul）+ `2·W_lm_head·B·1`（**只对最后一个位置**
      算 logits）+ `4·B·h·d_h·T²·L`（QK^T 与 AV）
    - 访存：权重读一遍（**B 个请求共享，不乘 B**）+ KV 写入 + 残差流读写
      +（非 flash）注意力分数读写 `O(T²)`

    ⚠️ `lm_head` 只算 `B·1` 不是 `B·T`：推理时只需要最后一个位置的 logits。
    写成 `B·T` 的话，与逐层实现的差是 `2·embed·B·(T-1)` —— 在 T=512 时
    **相对误差 27%**，正好是互校第一次跑出来的那个数。
    """
    wb = int(quant["weight_bits"]) / 8.0
    w_mat = _w_mat(model)
    h, dh, L = model.n_heads, model.head_dim, model.n_layers

    flops = 2.0 * (w_mat - model.embed_size) * batch * prompt   # 主干
    flops += 2.0 * model.embed_size * batch                     # lm_head：最后一位
    flops += 4.0 * batch * h * dh * prompt * prompt * L

    weights = w_mat * wb
    kv = batch * prompt * model.kv_bytes_per_token(int(quant["kv_bits"]))
    act = 2.0 * batch * prompt * model.hidden * ACT_BYTES * L
    if not backend["flash"]:
        act += 2.0 * batch * h * prompt * prompt * ACT_BYTES * L
    return Work(flops, weights + kv + act, weights, kv, act)


def analytic_decode(model: ModelSpec, backend: dict[str, Any],
                    quant: dict[str, Any], batch: int, context: int) -> Work:
    """Decode **一步**（每个请求产 1 个 token）。

    - 算力 `2·W_mat·B`（T = 1）+ `4·B·h·d_h·C·L`（注意力要看全部历史）
    - 访存：权重读一遍 + **KV 读 `B·C·kv_per_token`** + 残差流读写
      ⇒ 「context 拉长 ⇒ TPOT 变慢」是从这里**算出来的**，不是硬编码的曲线。
    """
    wb = int(quant["weight_bits"]) / 8.0
    w_mat = _w_mat(model)
    h, dh, L = model.n_heads, model.head_dim, model.n_layers

    flops = 2.0 * w_mat * batch
    flops += 4.0 * batch * h * dh * context * L

    weights = w_mat * wb
    kv = batch * context * model.kv_bytes_per_token(int(quant["kv_bits"]))
    act = 2.0 * batch * model.hidden * ACT_BYTES * L
    if not backend["flash"]:
        act += 2.0 * batch * h * context * ACT_BYTES * L
    return Work(flops, weights + kv + act, weights, kv, act)


# --------------------------------------------------------------------- 逐层
def layerwise_prefill(model: ModelSpec, backend: dict[str, Any],
                      quant: dict[str, Any], batch: int, prompt: int) -> Work:
    """与 `analytic_prefill` 同一件事，**逐层**算一遍。

    ⚠️ 这里**故意**不复用 `params_per_layer`：它把五个矩阵合成一个数，
    合成之后「漏掉了 o_proj」在闭式里根本看不出来。
    """
    wb = int(quant["weight_bits"]) / 8.0
    kv_bits = int(quant["kv_bits"])
    h, dh, L = model.n_heads, model.head_dim, model.n_layers
    d = model.hidden

    qkv = model.q_size + 2 * model.kv_size
    flops = weights = kv_b = act_b = 0.0
    for _ in range(L):
        flops += 2.0 * qkv * batch * prompt
        weights += qkv * wb
        flops += 4.0 * batch * h * dh * prompt * prompt
        flops += 2.0 * model.o_size * batch * prompt
        weights += model.o_size * wb
        flops += 2.0 * model.ffn_size * batch * prompt
        weights += model.ffn_size * wb
        act_b += 2.0 * batch * prompt * d * ACT_BYTES
        kv_b += batch * prompt * (2 * model.n_kv_heads * dh * kv_bits / 8.0)
        if not backend["flash"]:
            act_b += 2.0 * batch * h * prompt * prompt * ACT_BYTES
    flops += 2.0 * model.embed_size * batch       # lm_head 只算最后一位
    weights += model.embed_size * wb
    return Work(flops, weights + kv_b + act_b, weights, kv_b, act_b)


def layerwise_decode(model: ModelSpec, backend: dict[str, Any],
                     quant: dict[str, Any], batch: int, context: int) -> Work:
    """与 `analytic_decode` 同一件事，**逐层**算一遍。"""
    wb = int(quant["weight_bits"]) / 8.0
    kv_bits = int(quant["kv_bits"])
    h, dh, L = model.n_heads, model.head_dim, model.n_layers
    d = model.hidden

    qkv = model.q_size + 2 * model.kv_size
    flops = weights = kv_b = act_b = 0.0
    for _ in range(L):
        flops += 2.0 * qkv * batch
        weights += qkv * wb
        flops += 4.0 * batch * h * dh * context
        flops += 2.0 * model.o_size * batch
        weights += model.o_size * wb
        flops += 2.0 * model.ffn_size * batch
        weights += model.ffn_size * wb
        act_b += 2.0 * batch * d * ACT_BYTES
        kv_b += batch * context * (2 * model.n_kv_heads * dh * kv_bits / 8.0)
        if not backend["flash"]:
            act_b += 2.0 * batch * h * context * ACT_BYTES
    flops += 2.0 * model.embed_size * batch
    weights += model.embed_size * wb
    return Work(flops, weights + kv_b + act_b, weights, kv_b, act_b)


# --------------------------------------------------------------------- 设备
def _eff(backend: dict[str, Any], quant: dict[str, Any], kind: str) -> float:
    """有效算力利用率 = 后端效率 × **量化的算力折扣**。

    ⚠️ 量化省的是**带宽**不是算力 —— 推理时要反量化，算力侧反而略慢。
    少了 `compute_eff_scale` 这一项，「量化越狠越好」会变成模型的**假设**
    而不是模型的**结论**。
    """
    base = float(backend["gpu_eff"]) if kind == "gpu" else float(backend["cpu_eff"])
    return base * float(quant["compute_eff_scale"])


def seconds_of(work: Work, backend: dict[str, Any], quant: dict[str, Any],
               hw: HardwareSpec, n_layers: int, tokens_moved: int = 1) -> Timing:
    """把**工作量**变成**时间**（两条实现共用这一份）。

    - `gpu` / `cpu`：roofline `max(flops/(peak·eff), bytes/bw) + 开销`
    - `hybrid`：按 `offload` 比例**拆成两半串行**（时间是**加**出来的，
      不是「取平均」），再加一次边界激活的 PCIe 往返
    """
    kind = str(backend["device"])
    kernels = int(backend.get("kernels_per_layer", 8)) * n_layers
    overhead = (float(backend["launch_overhead_us"]) * 1e-6 * kernels
                + float(backend.get("scheduler_overhead_ms", 0.0)) * 1e-3)

    if kind == "gpu":
        if not hw.has_gpu:
            raise DeviceError(f"后端需要 GPU，但 {hw.name} 没有")
        comp = work.flops / max(hw.gpu_peak_flops * _eff(backend, quant, "gpu"), 1e-9)
        mem = work.bytes_ / max(hw.gpu_bw, 1e-9)
        return Timing(max(comp, mem) + overhead,
                      "compute" if comp >= mem else "memory",
                      comp, mem, overhead, 0.0)

    if kind == "cpu":
        comp = work.flops / max(hw.cpu_peak_flops * _eff(backend, quant, "cpu"), 1e-9)
        mem = work.bytes_ / max(hw.ram_bw, 1e-9)
        return Timing(max(comp, mem) + overhead,
                      "compute" if comp >= mem else "memory",
                      comp, mem, overhead, 0.0)

    if kind == "hybrid":
        if not hw.has_gpu:
            raise DeviceError(f"后端需要 GPU，但 {hw.name} 没有")
        off = float(backend["offload"])
        comp = (off * work.flops / max(hw.gpu_peak_flops * _eff(backend, quant, "gpu"), 1e-9)
                + (1 - off) * work.flops / max(hw.cpu_peak_flops * _eff(backend, quant, "cpu"), 1e-9))
        mem = (off * work.bytes_ / max(hw.gpu_bw, 1e-9)
               + (1 - off) * work.bytes_ / max(hw.ram_bw, 1e-9))
        xfer = 2.0 * tokens_moved * ACT_BYTES / max(hw.pcie_bw, 1e-9)
        return Timing(max(comp, mem) + xfer + overhead,
                      "compute" if comp >= mem else "memory",
                      comp, mem, overhead, xfer)

    raise DeviceError(f"未知 device：{kind}")


# --------------------------------------------------------------------- 拐点
def arithmetic_intensity(flops: float, bytes_: float) -> float:
    return flops / bytes_ if bytes_ > 0 else float("inf")


def human_bytes(n: float) -> str:
    unit = "B"
    for u in ("B", "KB", "MB", "GB", "TB"):
        unit = u
        if abs(n) < 1024.0 or u == "TB":
            break
        n /= 1024.0
    return f"{n:.2f} {unit}"
