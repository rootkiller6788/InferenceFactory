"""设计空间：**笛卡尔积全枚举**（不是搜索）。

⚠️ 因为是全枚举，验收的第一条是「**矩阵缺口 = 0**」，不是「增益 ≥ x」
（`L5-quantfactory` 里也是这条）。搜索器才有「找到的最好有多好」这个问题。
"""

from __future__ import annotations

import itertools
from typing import Any

from .specs import HardwareSpec, ModelSpec

#: 配置轴的顺序（`config_id` / `describe` 都按它，改了会换掉所有 id）
AXIS_ORDER: tuple[str, ...] = ("backend", "quant", "batch", "context", "arrival")


def enumerate_configs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """`axes` 的笛卡尔积，顺序**确定**（按 `AXIS_ORDER`、按列表顺序）。"""
    axes = cfg["axes"]
    keys = [k for k in AXIS_ORDER if k in axes]
    values = [axes[k] for k in keys]
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*values):
        out.append(dict(zip(keys, combo)))
    return out


def config_id(c: dict[str, Any]) -> str:
    a = float(c["arrival"])
    return (f"c:{c['backend']}|{c['quant']}|b{c['batch']}"
            f"|c{c['context']}|a{a:g}")


def describe(c: dict[str, Any]) -> str:
    a = float(c["arrival"])
    return (f"{c['backend']} / {c['quant']} / batch={c['batch']} / "
            f"ctx={c['context']} / {a:g} req/s")


def deployment_id(model: ModelSpec, hw: HardwareSpec) -> str:
    return f"{model.name}@{hw.name}"


def enumerate_deployments(models: tuple[ModelSpec, ...],
                          hardware: tuple[HardwareSpec, ...],
                          ) -> list[tuple[ModelSpec, HardwareSpec]]:
    return [(m, h) for m in models for h in hardware]


def matrix_expected(models: tuple[ModelSpec, ...],
                    hardware: tuple[HardwareSpec, ...],
                    cfg: dict[str, Any]) -> int:
    return len(models) * len(hardware) * len(enumerate_configs(cfg))
