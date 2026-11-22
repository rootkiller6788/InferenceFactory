"""三份 spec：**模型 / 硬件**（数据）+ **后端**（配置，见 `config.backends`）。

⚠️ **参数量是算出来的，不是写在数据文件里的。** 写了就会和「结构」不一致 ——
改一个 `n_layers` 而忘了改 `params` 是最典型的那种 bug：代码照跑、
数字看着也对，只是**那一行不再描述这个模型了**。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

#: 项目根：`src/inference/specs.py` → 上两级。
_PKG_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSpec:
    """一个 dense Transformer 解码器（无 MoE）。"""

    name: str
    family: str
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_mult: float
    vocab: int
    tie_embeddings: bool
    kv_dtype_bits: int

    # ---------------------------------------------------------------- 结构
    # ⚠️ 全用 `cached_property`：这些值只由**不可变字段**决定，而
    #    `layerwise_*` 的逐层循环会反复读 `ffn_size` / `params_per_layer`
    #    （互校 891 组 × 层数）。重算一次是乘法，重算十万次就是剖面里的热点。
    @cached_property
    def ffn_hidden(self) -> int:
        # ⚠️ `round` 而不是 `int(...)`：SwiGLU 的 intermediate size 通常是
        #    先乘后取到 256 的倍数。这里取 `round` 保证**整数且确定**
        #    （浮点 `ffn_mult * hidden` 直接 `int()` 会因舍入方向丢 1）。
        return int(round(self.ffn_mult * self.hidden))

    @cached_property
    def q_size(self) -> int:
        return self.hidden * (self.n_heads * self.head_dim)

    @cached_property
    def kv_size(self) -> int:
        return self.hidden * (self.n_kv_heads * self.head_dim)

    @cached_property
    def o_size(self) -> int:
        return (self.n_heads * self.head_dim) * self.hidden

    @cached_property
    def ffn_size(self) -> int:
        # SwiGLU：gate / up / down 三个矩阵
        return 3 * self.hidden * self.ffn_hidden

    @cached_property
    def embed_size(self) -> int:
        return self.vocab * self.hidden

    @cached_property
    def params_per_layer(self) -> int:
        return self.q_size + 2 * self.kv_size + self.o_size + self.ffn_size

    @cached_property
    def params(self) -> int:
        head = 0 if self.tie_embeddings else self.embed_size
        return self.n_layers * self.params_per_layer + self.embed_size + head

    @cached_property
    def active_params(self) -> int:
        """参与前向的权重数（dense ⇒ 全部）。"""
        return self.params

    # ---------------------------------------------------------------- 体积
    def weight_bytes(self, bits: int) -> float:
        return self.params * bits / 8.0

    def kv_bytes_per_token(self, kv_bits: int) -> float:
        """每 token 的 KV cache 字节数（K 与 V 两份）。"""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * kv_bits / 8.0

    # ---------------------------------------------------------------- 序列化
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "n_layers": self.n_layers,
            "hidden": self.hidden,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "ffn_mult": self.ffn_mult,
            "ffn_hidden": self.ffn_hidden,
            "vocab": self.vocab,
            "tie_embeddings": self.tie_embeddings,
            "kv_dtype_bits": self.kv_dtype_bits,
            "params": self.params,
            "params_per_layer": self.params_per_layer,
            "params_b": self.params / 1e9,
            "weight_bytes": {b: self.weight_bytes(b) for b in (16, 8, 4)},
            "kv_bytes_per_token": self.kv_bytes_per_token(self.kv_dtype_bits),
            "gqa_ratio": self.n_heads / self.n_kv_heads,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ModelSpec":
        return ModelSpec(
            name=str(d["name"]),
            family=str(d.get("family", "llama")),
            n_layers=int(d["n_layers"]),
            hidden=int(d["hidden"]),
            n_heads=int(d["n_heads"]),
            n_kv_heads=int(d["n_kv_heads"]),
            head_dim=int(d["head_dim"]),
            ffn_mult=float(d["ffn_mult"]),
            vocab=int(d["vocab"]),
            tie_embeddings=bool(d.get("tie_embeddings", True)),
            kv_dtype_bits=int(d.get("kv_dtype_bits", 16)),
        )


@dataclass(frozen=True)
class HardwareSpec:
    """一台机器：GPU（可选）+ CPU + 内存。"""

    name: str
    has_gpu: bool
    gpu_peak_flops: float   # FLOP/s（fp16 tensor core）
    gpu_bytes: float        # 总显存
    gpu_reserved_bytes: float  # 驱动/显示占用，不属于推理可用
    gpu_bw: float           # B/s
    pcie_bw: float          # B/s（GPU ↔ 主机）
    cpu_peak_flops: float
    ram_bytes: float
    ram_bw: float

    @property
    def gpu_usable_bytes(self) -> float:
        return self.gpu_bytes - self.gpu_reserved_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "has_gpu": self.has_gpu,
            "gpu_peak_flops": self.gpu_peak_flops,
            "gpu_bytes": self.gpu_bytes,
            "gpu_reserved_bytes": self.gpu_reserved_bytes,
            "gpu_usable_bytes": self.gpu_usable_bytes,
            "gpu_bw": self.gpu_bw,
            "pcie_bw": self.pcie_bw,
            "cpu_peak_flops": self.cpu_peak_flops,
            "ram_bytes": self.ram_bytes,
            "ram_bw": self.ram_bw,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "HardwareSpec":
        has = bool(d["has_gpu"])
        return HardwareSpec(
            name=str(d["name"]),
            has_gpu=has,
            gpu_peak_flops=float(d.get("gpu_peak_flops", 0.0)),
            gpu_bytes=float(d.get("gpu_bytes", 0.0)),
            gpu_reserved_bytes=float(d.get("gpu_reserved_bytes", 0.0)),
            gpu_bw=float(d.get("gpu_bw", 0.0)),
            pcie_bw=float(d.get("pcie_bw", 0.0)),
            cpu_peak_flops=float(d["cpu_peak_flops"]),
            ram_bytes=float(d["ram_bytes"]),
            ram_bw=float(d["ram_bw"]),
        )


def _load_dir(directory: Path, cls: Any) -> list[Any]:
    if not directory.exists():
        raise FileNotFoundError(f"数据目录不存在：{directory}")
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"数据目录里没有 .json：{directory}")
    return [cls.from_dict(json.loads(f.read_text(encoding="utf-8"))) for f in files]


def load_models(cfg: dict[str, Any], root: Path | None = None) -> tuple[ModelSpec, ...]:
    d = Path(cfg["data"]["models_dir"])
    if not d.is_absolute():
        d = (root or _PKG_ROOT) / d
    return tuple(_load_dir(d, ModelSpec))


def load_hardware(cfg: dict[str, Any], root: Path | None = None) -> tuple[HardwareSpec, ...]:
    d = Path(cfg["data"]["hardware_dir"])
    if not d.is_absolute():
        d = (root or _PKG_ROOT) / d
    return tuple(_load_dir(d, HardwareSpec))
