"""配置校验：坏配置要在**入口**就被拒绝，并给出可读原因。

⚠️ 自包含：不依赖 conftest 夹具（与 `test_import.py` / `test_cli_contract.py`
一致）—— 判据的唯一真源是 `diagnostics/`，这里只锁新增的校验行为。
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.config import (  # noqa: E402
    DEFAULTS, ConfigError, load_config, validate_config,
)


def test_default_config_is_valid() -> None:
    validate_config(copy.deepcopy(DEFAULTS))
    assert load_config(ROOT / "configs" / "default.json", root=ROOT)


def test_broken_json_becomes_config_error(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text('{"axes": ', encoding="utf-8")
    with pytest.raises(ConfigError, match="不是合法 JSON"):
        load_config(p)


def test_non_object_config_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "list.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ConfigError, match="顶层必须是对象"):
        load_config(p)


@pytest.mark.parametrize("axis,value", [
    ("batch", "8"),
    ("batch", 0),
    ("context", 1.5),
    ("arrival", 0),
    ("arrival", float("inf")),
])
def test_axis_values_are_type_checked(axis: str, value: object) -> None:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["axes"][axis] = [value]
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_missing_axis_is_rejected() -> None:
    cfg = copy.deepcopy(DEFAULTS)
    del cfg["axes"]["arrival"]
    with pytest.raises(ConfigError, match="缺少设计轴"):
        validate_config(cfg)
