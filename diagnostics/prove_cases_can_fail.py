"""证明**每一条**对抗用例都能红。

⚠️ 为什么必须有这个脚本：一条恒真的判据与一条真的判据在产物里长得**一模一样**
（都是 `held: true`）。光看报告分不出「这里验过了」与「这里什么都没验」。
唯一可靠的发现手段是：**逐条造一个能让它红的输入**。

两段式：
  ① **通用变异** —— 与内容无关的破坏（删产物、改数、截列表、反转、改图/改配置）。
     只作用在该用例**真的读过**的产物上（探针 dict 记下访问键）：
     不限定靶的话，2 万条变异池里每条用例只试得动几十条，
     **打不中靶会被误判成「用例恒真」**。
  ② **定向变异** —— 通用变异打不动的给一条手工构造的、带理由的破坏，
     每条都要说清「对应哪种真缺陷」。

结论落 `diagnostics/mutation.json` 供 `check_readme.py` 引用。
"""

from __future__ import annotations

import copy
import importlib
import json
import pathlib
import random
import re
import sys
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inference import invariants as INV          # noqa: E402
from inference import specs as SP                # noqa: E402
from inference.config import load_config         # noqa: E402

MAX_MUT = 60          # 每条用例最多试多少个通用变异
SEED = 20261003

#: 全部产物的文件名（含点），路径解析要先按它们剥前缀。
_KNOWN_PRODUCTS: tuple[str, ...] = tuple(sorted(
    p.name for p in (ROOT / "out").glob("*.json")))


# --------------------------------------------------------------------- 探针
class Probe(dict):
    """记下**被读过**的键路径的 dict。

    ⚠️ **所有层的子探针共享同一个 `touched` 列表**（用 `register()` 传进来）。
    第一版每层 `Probe(v)` 都新建一个 `self.touched = []` ⇒ 记录散在
    N 个子实例里，而外面只看**根实例**的那一份 ⇒ 每个用例都只有 1 条记录，
    通用变异找不到靶 ⇒ 全部判成「打不中」，而「打不中」会被误读成
    **「用例恒真」**。踩这个坑花的力气比写这个脚本还多 ——
    「探针记到的东西」本身就是一个必须核对的对象。
    """

    def __init__(self, *a, touched: list[str] | None = None, path: str = "", **kw):
        super().__init__(*a, **kw)
        #: 共享的访问记录（**同一个列表对象**，不是副本）
        self.touched: list[str] = touched if touched is not None else []
        self.path: str = path

    def _note(self, key):
        self.touched.append(f"{self.path}.{key}" if self.path else str(key))

    def __getitem__(self, k):
        self._note(k)
        v = super().__getitem__(k)
        return _wrap_probe(v, f"{self.path}.{k}" if self.path else str(k), self.touched)

    def get(self, k, default=None):
        self._note(k)
        v = super().get(k, default)
        return _wrap_probe(v, f"{self.path}.{k}" if self.path else str(k), self.touched)

    def items(self):
        out = []
        for k in list(super().__iter__()):
            self._note(k)
            v = super().__getitem__(k)
            out.append((k, _wrap_probe(v, f"{self.path}.{k}" if self.path else str(k),
                                       self.touched)))
        return out

    def values(self):
        return [v for _, v in self.items()]


def _wrap_probe(v, path, touched):
    if isinstance(v, dict) and not isinstance(v, Probe):
        return Probe(v, touched=touched, path=path)
    if isinstance(v, list):
        return [_wrap_probe(x, f"{path}[{i}]", touched) for i, x in enumerate(v)]
    return v


# ----------------------------------------------------------------- 收集数字
def walk_json_nums(obj, path="", out=None):
    if out is None:
        out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk_json_nums(v, f"{path}.{k}" if path else k, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk_json_nums(v, f"{path}[{i}]", out)
    elif isinstance(obj, bool):
        # ⚠️ `bool` 必须排在 `int` 前面：`True` 也是 `int` 的实例。
        pass
    elif isinstance(obj, int):
        out.append((path, obj))          # 值，不是类型（曾写成 `int` ⇒ abs(type) 崩）
    elif isinstance(obj, float):
        out.append((path, obj))
    return out


def _tokens(path: str) -> list[str]:
    """把 `matrix.json.cells[0].readings.p99_ms` 切成
    `["matrix.json", "cells", "[0]", "readings", "p99_ms"]`。

    ⚠️ 产物名本身含点（`matrix.json`）⇒ 不能无脑按 `.` 切。
    先按**已知产物名**剥前缀，再切剩下的。
    """
    for name in _KNOWN_PRODUCTS:
        if path == name:
            return [name]
        if path.startswith(name + "."):
            rest = path[len(name) + 1:]
            toks = [name]
            for chunk in rest.split("."):
                if not chunk:
                    continue
                # `cells[0]` -> `cells`, `[0]`
                base, *rest_chunks = re.split(r"(\[[0-9]+\])", chunk)
                if base:
                    toks.append(base)
                toks.extend(x for x in rest_chunks if x)
            return toks
    return path.split(".")


def set_by_path(root, path, value):
    """按 `matrix.json.cells[0].readings.p99_ms` 路径写回。"""
    toks = _tokens(path)
    if not toks:
        return False
    cur = root
    for tok in toks[1:-1]:
        cur = _step(cur, tok)
        if cur is None:
            return False
    last = toks[-1]
    if last.startswith("[") and last.endswith("]"):
        i = int(last[1:-1])
        if not isinstance(cur, list) or i >= len(cur):
            return False
        cur[i] = value
        return True
    if not isinstance(cur, dict) or last not in cur:
        return False
    cur[last] = value
    return True


def get_by_path(root, path):
    cur = root
    for tok in _tokens(path)[1:]:
        cur = _step(cur, tok)
        if cur is None:
            return None
    return cur


def _step(cur, tok):
    try:
        if tok.startswith("[") and tok.endswith("]"):
            i = int(tok[1:-1])
            if not isinstance(cur, list) or i >= len(cur):
                return None
            return cur[i]
        if not isinstance(cur, dict) or tok not in cur:
            return None
        return cur[tok]
    except Exception:                                    # noqa: BLE001
        return None


# --------------------------------------------------------------- 通用变异池
def product_names(products: dict[str, Any]) -> list[str]:
    return [k for k in sorted(products) if k.endswith(".json")]


def generic_mutations(products: dict[str, Any], touched: set[str],
                      rng: random.Random) -> list[tuple[str, dict[str, Any]]]:
    """与内容无关的破坏。`touched` 是「这条用例真的读过的产物」全集。

    顺序有讲究：**内容变异排在「删产物」之前**。
    删产物只能证明「用例读了这份产物」，证明不了它在看**值** ——
    一个把产物读了却不检查内容的用例，删产物也照样红。
    """
    muts: list[tuple[str, dict[str, Any]]] = []
    # ⚠️ 产物名含点（`matrix.json`）⇒ 不能按 `.` 切一层就取；必须先匹配**已知产物名**。
    #    第一版写成 `t.split(".")[0]` ⇒ 全变成 `matrix` / `compare`
    #    ⇒ 与 `products` 的键（`matrix.json`）对不上 ⇒ 靶集恒空、变异池恒 0，
    #    而「池为 0」被静默当成「变异都没打中」，最后误报成「用例恒真」。
    touched_files = {name for name in products
                     if any(t == name or t.startswith(name + ".")
                            for t in touched)}

    # ---- ① 内容：按路径改数 -------------------------------------------
    pool: list[tuple[str, str, Any]] = []      # (产物, 路径, 新值)
    for name in sorted(touched_files):
        for p, v in walk_json_nums(products[name], name):
            if isinstance(v, int) and not isinstance(v, bool):
                pool.append((name, p, 0 if v else 1))
                pool.append((name, p, v * 2 + 1))
            else:
                pool.append((name, p, 0.0 if v else 1.0))
                pool.append((name, p, -abs(v) - 1.0))
    rng.shuffle(pool)
    for name, p, val in pool[:MAX_MUT]:
        m = copy.deepcopy(products)
        if set_by_path(m[name], p, val):
            muts.append((f"改数 {name}:{p} -> {val!r}", m))

    # ---- ② 截断 / 反转列表 ---------------------------------------------
    lists: list[tuple[str, str, int]] = []

    def collect_lists(obj, name, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                collect_lists(v, name, f"{path}.{k}" if path else k)
        elif isinstance(obj, list) and obj:
            lists.append((name, path, len(obj)))
            for i, v in enumerate(obj):
                collect_lists(v, name, f"{path}[{i}]")

    for name in sorted(touched_files):
        collect_lists(products[name], name)
    rng.shuffle(lists)
    for name, p, n in lists[:MAX_MUT]:
        if n < 2:
            continue
        # 截掉一半（至少留 1）
        m = copy.deepcopy(products)
        arr = get_by_path(m[name], p)
        if isinstance(arr, list) and len(arr) >= 2:
            del arr[len(arr) // 2:]
            muts.append((f"截半 {name}:{p} ({n} -> {len(arr)})", m))
        # 反转
        m2 = copy.deepcopy(products)
        arr2 = get_by_path(m2[name], p)
        if isinstance(arr2, list) and len(arr2) >= 2:
            arr2.reverse()
            muts.append((f"反转 {name}:{p}", m2))

    # ---- ③ 布尔翻面 ----------------------------------------------------
    bools: list[tuple[str, str, bool]] = []

    def collect_bools(obj, name, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                collect_bools(v, name, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                collect_bools(v, name, f"{path}[{i}]")
        elif isinstance(obj, bool):
            bools.append((name, path, obj))

    for name in sorted(touched_files):
        collect_bools(products[name], name)
    rng.shuffle(bools)
    for name, p, v in bools[:MAX_MUT]:
        m = copy.deepcopy(products)
        if set_by_path(m[name], p, not v):
            muts.append((f"翻面 {name}:{p} -> {not v}", m))

    # ---- ④ 删产物（**排最后**）-----------------------------------------
    for name in sorted(touched_files):
        m = copy.deepcopy(products)
        del m[name]
        muts.append((f"删产物 {name}", m))

    return muts[:MAX_MUT * 4]


# --------------------------------------------------------------- 定向变异
def cfg_mutations(cfg: dict[str, Any], touched: set[str]) -> list[tuple[str, dict]]:
    """**变异配置对象**。

    ⚠️ 有一批用例根本不读产物 `p`，读的是 `c`（配置）与 `ms`（模型）——
    比如「每根轴非空无重复」「后端都在目录里」「指纹是口径的函数」。
    只变异 `p` 的话，这些用例**永远**打不动，然后被误判成「恒真」。
    靶必须覆盖**用例真的读过的每一个入参**，不只是产物。

    ⚠️ 而且**不能**拿 `touched` 去筛这些变异：探针只包了 `p`，
    读 `c` 的用例 `touched` 是空的 ⇒ 用一个空的靶集去筛 ⇒ 一条变异都生成不出来
    （第一版就是这么写的，结果这 5 条一条都没打红）。
    所以这里的每一条都是**无条件**生成。
    """
    muts: list[tuple[str, dict]] = []
    for axis in (cfg.get("axes") or {}):
        m = copy.deepcopy(cfg)
        m["axes"][axis] = []
        muts.append((f"清空轴 cfg.axes.{axis}", m))
        m2 = copy.deepcopy(cfg)
        v = list(m2["axes"][axis])
        if v:
            m2["axes"][axis] = v + [v[0]]
            muts.append((f"给 cfg.axes.{axis} 塞一个重复值", m2))

    for axis, catalog in (("backend", "backends"), ("quant", "quant")):
        if axis in (cfg.get("axes") or {}) and isinstance(cfg.get(catalog), dict):
            m = copy.deepcopy(cfg)
            m[catalog]["幽灵条目"] = dict(next(iter(m[catalog].values())))
            muts.append((f"往 cfg.{catalog} 加一个轴里没有的键", m))
            m2 = copy.deepcopy(cfg)
            live = m2["axes"][axis][0]
            if live in m2[catalog]:
                del m2[catalog][live]
            muts.append((f"从 cfg.{catalog} 删掉轴里有的 `{live}`", m2))

    # 指纹：改一个**不参与**协议的字段 —— 看指纹会不会被无关字段带走
    m = copy.deepcopy(cfg)
    m["seed"] = int(m.get("seed", 0)) + 1
    muts.append(("改 cfg.seed（看指纹是否被无关字段带走）", m))

    # 量化位宽：让「位宽越大体积越大」不成立
    if isinstance(cfg.get("quant"), dict):
        for qname, q in cfg["quant"].items():
            m = copy.deepcopy(cfg)
            m["quant"][qname]["weight_bits"] = int(q.get("weight_bits", 16)) * 3 + 1
            muts.append((f"把 cfg.quant.{qname}.weight_bits 抬到反常值", m))

    if isinstance(cfg.get("workload"), dict):
        m = copy.deepcopy(cfg)
        m["workload"]["prompt_tokens"] = int(m["workload"]["prompt_tokens"]) * 3
        muts.append(("把 prompt_tokens 乘 3（看两条成本路径是否还一致）", m))
    return muts


def src_patches() -> list[tuple[str, Callable[[], Any]]]:
    """**打源码补丁**：把某个纯函数改坏，看用例是否红。

    ⚠️ 有一批用例读的是**源码里的纯函数**（Erlang-B/C、闭式 vs 逐层），
    产物里没有任何东西能让它们红 —— 唯一的靶就是**函数本身**。
    `gen()` 是生成器：`next` 一次装补丁，`next` 第二次还原（`finally` 保证）。
    """
    from inference import cost as C
    from inference import queueing as Q
    from inference import config as CF

    def _swap(mod, name, bad):
        def gen():
            orig = getattr(mod, name)
            setattr(mod, name, bad)
            yield
            setattr(mod, name, orig)
        return gen

    return [
        ("把 Erlang-B 改成常数 1.7（= 概率跑到 [0,1] 外）",
         _swap(Q, "erlang_b", lambda c, a: 1.7)),
        ("把 Erlang-C 压到 Erlang-B 的一半（= 等待概率比阻塞概率还低）",
         _swap(Q, "erlang_c", lambda c, a: Q.erlang_b(c, a) * 0.5 - 1e-9)),
        ("让逐层累加的**访存量**与闭式差 1（= 两条独立实现漂移）",
         _swap(C, "layerwise_prefill",
               lambda m, b, q, n, T: _bumped(
                   C.analytic_prefill(m, b, q, n, T)))),
        ("让逐层累加的 **flops** 与闭式差 1（= 两条独立实现漂移）",
         _swap(C, "layerwise_decode",
               lambda m, b, q, n, T: _bumped(
                   C.analytic_decode(m, b, q, n, T)))),
        # ⚠️ 下面两条的靶**只能是源码**：它们读的是纯函数本身，
        #    产物与配置里没有任何东西能让它们红。
        ("让协议指纹每次都不一样（= 指纹不是口径的函数）",
         _swap(CF, "protocol_digest", lambda c: f"unstable-{id(c)}")),
        # `ModelSpec` 是 `frozen=True` 的 dataclass，但 **类级** 换方法是允许的：
        # `frozen` 拦的是**实例属性赋值**，不拦类字典。用例里全是 `m.weight_bytes(bits)`。
        # ⚠️ 破坏必须是**反单调**，不能是「换成常数」：判据是「非降」，
        #    常数并不违反非降（相等的比较是允许的）⇒ 常数打不红是**判据正确**，
        #    不是判据弱。挑破坏的时候要顺着判据的方向挑。
        ("让权重占用随位宽**变大而变小**（= 反单调）",
         _swap_class(SP.ModelSpec, "weight_bytes",
                     lambda self, bits: self.params * 8.0 / max(bits, 1))),
    ]


def _swap_class(cls, name, bad):
    def gen():
        orig = getattr(cls, name)
        setattr(cls, name, bad)
        yield
        setattr(cls, name, orig)
    return gen


def _bumped(work):
    """返回一个**只差 1 字节 / 1 flop** 的 Work —— 相对误差仍然 > 1e-12。"""
    return type(work)(flops=work.flops + 1, bytes_=work.bytes_ + 1,
                      weights_bytes=work.weights_bytes,
                      kv_bytes=work.kv_bytes, act_bytes=work.act_bytes)


def targeted_mutations(products, cfg, ms, hs):
    """通用变异打不动的用例，给一条手工构造的破坏（每条写清对应哪种真缺陷）。"""
    T: dict[str, list[tuple[str, Callable[[], Any]]]] = {}

    def add(name, why, fn):
        T.setdefault(name, []).append((why, fn))

    def mut(fn):
        def inner():
            m = copy.deepcopy(products)
            fn(m)
            return m
        return inner

    # ---- 不可行格也要带占用明细（本轮真修过的一条）----
    def _strip_infeasible_memory(m):
        for c in m["matrix.json"]["cells"]:
            if not c["feasible"]:
                c.pop("memory", None)
    add("infeasible_cells_still_carry_a_memory_breakdown",
        "不可行格不带 memory（= 修之前的样子：『为什么不可行』只剩一个 reason 字符串）",
        mut(_strip_infeasible_memory))

    # ---- 版本清单与实际产物一致 ----
    def _drop_version_file(m):
        v = m["version.json"]
        if v["versioned_files"]:
            v["versioned_files"] = v["versioned_files"][:-1]
    add("version_digest_covers_every_declared_file",
        "版本清单里删掉一个文件（= 声明的文件集与实际产物漂移）",
        mut(_drop_version_file))

    # ---- 决策必须满足 SLO ----
    def _break_decision_slo(m):
        for d in m["compare.json"]["per_deployment"]:
            if d["decision"]:
                cell = None
                for c in m["matrix.json"]["cells"]:
                    if (c["deployment"] == d["deployment"]
                            and c["backend"] == d["decision"]["backend"]
                            and c["quant"] == d["decision"]["quant"]
                            and c["batch"] == d["decision"]["batch"]
                            and c["context"] == d["decision"]["context"]
                            and c["arrival_rate"] == d["decision"]["arrival_rate"]):
                        cell = c
                        break
                if cell is not None and cell["feasible"]:
                    d["decision"]["readings"]["p99_ms"] = 10.0 ** 6
                return
    add("every_decision_meets_the_slo",
        "把一个部署决策的 P99 改到 SLO 之外（= 选了个过不了 SLO 的格子）",
        mut(_break_decision_slo))

    def _break_decision_feasible(m):
        for d in m["compare.json"]["per_deployment"]:
            if d["decision"]:
                d["decision"]["backend"] = "不存在的后端"
                return
    add("every_decision_is_a_feasible_cell",
        "把决策指到一个矩阵里没有的后端（= 决策不是从矩阵里选的）",
        mut(_break_decision_feasible))

    # ---- 最优解就是极值 ----
    def _break_best_throughput(m):
        for d in m["compare.json"]["per_deployment"]:
            b = d.get("best_throughput")
            if b:
                b["readings"]["tokens_per_s"] += 12345.0
                return
    add("best_throughput_is_the_max_over_feasible_cells",
        "把「最吞吐」的读数抬高一点（= 报出来的不是真正的最大值）",
        mut(_break_best_throughput))

    def _break_best_p99(m):
        for d in m["compare.json"]["per_deployment"]:
            b = d.get("best_p99") or d.get("best_latency")
            if b:
                b["readings"]["p99_ms"] -= 777.0
                return
    add("best_p99_is_the_min_over_feasible_cells",
        "把「最优 P99」压到比真正的最小值还低（= 前沿点不在前沿上）",
        mut(_break_best_p99))

    def _inject_dominated(m):
        pg = m["compare.json"]["per_deployment"]
        src = None
        for d in pg:
            if d["n_pareto"] and d.get("pareto"):
                src = d
                break
        if src is None:
            return
        victim = next((c for c in m["matrix.json"]["cells"]
                       if c["feasible"] and c["deployment"] == src["deployment"]
                       and not any(c is q for q in src["pareto"])), None)
        if victim is not None:
            src["pareto"].append(victim)
    add("pareto_front_contains_no_dominated_cell",
        "往前沿里塞一个被支配的格子（= 前沿退化成「所有可行格」）",
        mut(_inject_dominated))

    # ---- 曲线轴 ----
    def _break_curve_x(m):
        for s in m["scaling.json"]["series"]:
            cv = s["curves"]["batch"]
            if len(cv) >= 2:
                cv[0]["x"], cv[1]["x"] = cv[1]["x"], cv[0]["x"]
                return
    add("curve_axes_are_strictly_increasing",
        "把曲线 x 轴相邻两点对调（= 画出来会打结）",
        mut(_break_curve_x))

    def _break_curve_n(m):
        for s in m["scaling.json"]["series"]:
            cv = s["curves"]["context"]
            if cv:
                cv.pop()
                return
    add("scaling_point_count_matches_the_declared_grids",
        "从某条曲线上删一个点（= 点数与声明的网格对不上）",
        mut(_break_curve_n))

    def _drop_curve(m):
        for s in m["scaling.json"]["series"]:
            s["curves"].pop("arrival", None)
            return
    add("every_series_has_all_three_curves",
        "删掉 arrival 曲线（= 少扫了一根轴却照样报「三条曲线」）",
        mut(_drop_curve))

    # ---- 单调性：改一格的反例 ----
    def _break_kv_ctx(m):
        cells = [c for c in m["matrix.json"]["cells"]
                 if c["feasible"] and c.get("memory", {}).get("kv_bytes")]
        if cells:
            cells[0]["memory"]["kv_bytes"] += 10 ** 9
    add("kv_bytes_is_monotone_in_context",
        "给某一格的 KV 占用加一个定值（= KV 随 context 非降被打破）",
        mut(_break_kv_ctx))

    def _break_tpot_ctx(m):
        for c in m["matrix.json"]["cells"]:
            r = c.get("readings") or {}
            if c["feasible"] and r.get("tpot_ms"):
                r["tpot_ms"] *= 0.5
                return
    add("tpot_is_monotone_in_context",
        "把某一格的 TPOT 砍半（= 上下文越长越快）",
        mut(_break_tpot_ctx))

    def _break_tpot_batch(m):
        for c in m["matrix.json"]["cells"]:
            r = c.get("readings") or {}
            if c["feasible"] and r.get("decode_s"):
                r["decode_s"] *= 0.25
                return
    add("decode_step_time_is_monotone_in_batch",
        "把某一格的 decode 步时间砍到 1/4（= batch 越大每步越快）",
        mut(_break_tpot_batch))

    def _break_rho(m):
        for c in m["matrix.json"]["cells"]:
            r = c.get("readings") or {}
            if c["feasible"] and r.get("rho"):
                r["rho"] = 0.99 - r["rho"]
                return
    add("utilization_is_monotone_in_arrival",
        "把某一格的 ρ 反射到 1 附近（= 利用率随负载非降被打破）",
        mut(_break_rho))

    def _break_p99_arr(m):
        for c in m["matrix.json"]["cells"]:
            r = c.get("readings") or {}
            if c["feasible"] and r.get("p99_ms"):
                r["p99_ms"] = 1.0
                return
    add("p99_is_monotone_in_arrival",
        "把某一格的 P99 压到 1 ms（= 负载越高延迟越低）",
        mut(_break_p99_arr))

    def _break_wbits(m):
        for c in m["matrix.json"]["cells"]:
            mm = c.get("memory") or {}
            if mm.get("weight_bytes"):
                mm["weight_bytes"] = 1
                return
    add("weight_bytes_is_monotone_in_quant_bits",
        "把某一格的权重占用压到 1 字节（= 位宽与权重占用脱钩）",
        mut(_break_wbits))

    # ---- 排队论 ----
    def _break_eb_unit(m):
        for s in m["scaling.json"]["series"]:
            for q in s["curves"]["arrival"]:
                if q.get("erlang_b") is not None:
                    q["erlang_b"] = 1.7
                    return
    add("erlang_b_is_in_the_unit_interval",
        "把某个 Erlang-B 值写成 1.7（= 概率跑到 [0,1] 外面）",
        mut(_break_eb_unit))

    def _break_ec_eb(m):
        for s in m["scaling.json"]["series"]:
            for q in s["curves"]["arrival"]:
                if (q.get("erlang_b") is not None
                        and q.get("erlang_c") is not None):
                    q["erlang_c"] = q["erlang_b"] * 0.5
                    return
    add("erlang_c_is_at_least_erlang_b",
        "把某个 Erlang-C 压到 Erlang-B 的一半（= 等待概率比阻塞概率还低）",
        mut(_break_ec_eb))

    def _unmark_unstable(m):
        for c in m["matrix.json"]["cells"]:
            r = c.get("readings") or {}
            if c["feasible"] and r.get("rho", 0) >= 1.0:
                return
        # 没有现成的：把某个可行格标成 ρ ≥ 1 但保留 feasible
        for c in m["matrix.json"]["cells"]:
            if c["feasible"] and c.get("readings"):
                c["readings"]["rho"] = 1.5
                return
    add("unstable_cells_are_marked_infeasible",
        "把某个可行格的 ρ 标成 1.5（= ρ ≥ 1 却没被判不可行）",
        mut(_unmark_unstable))

    def _break_p99_ge_service(m):
        for c in m["matrix.json"]["cells"]:
            r = c.get("readings") or {}
            if c["feasible"] and r.get("p99_ms") and r.get("service_s"):
                r["p99_ms"] = r["service_s"] * 1e3 * 0.1
                return
    add("p99_covers_at_least_the_service_time",
        "把某一格的 P99 压到服务时间的 1/10（= 延迟低于它自己的服务时间）",
        mut(_break_p99_ge_service))

    def _break_wait_cap(m):
        for c in m["matrix.json"]["cells"]:
            r = c.get("readings") or {}
            if c["feasible"] and r.get("wait_batch_s"):
                r["wait_batch_s"] = 10.0 ** 6
                return
    add("static_batching_wait_is_capped",
        "把某一格的凑批等待改成 1e6 秒（= 静态批无限期等待）",
        mut(_break_wait_cap))

    def _break_eb_cap(m):
        for c in m["matrix.json"]["cells"]:
            r = c.get("readings") or {}
            if c["feasible"] and r.get("effective_batch"):
                r["effective_batch"] = c["batch"] + 1000
                return
    add("effective_batch_never_exceeds_the_configured_batch",
        "把某一格的实际 batch 抬到配置值之上（= 批里凭空多出请求）",
        mut(_break_eb_cap))

    # ---- 内存 ----
    def _break_down_batch(m):
        for c in m["matrix.json"]["cells"]:
            if c["feasible"] and c["batch"] == 1:
                c["feasible"] = False
                c["reason"] = "gpu_oom"
                return
    add("memory_feasibility_is_downward_closed_in_batch",
        "把一个 batch=1 的可行格判成 OOM（= 更小的批反而放不下）",
        mut(_break_down_batch))

    def _break_down_ctx(m):
        for c in m["matrix.json"]["cells"]:
            if c["feasible"] and c["context"] == 2048:
                c["feasible"] = False
                c["reason"] = "gpu_oom"
                return
    add("memory_feasibility_is_downward_closed_in_context",
        "把一个 ctx=2048 的可行格判成 OOM（= 更短的上下文反而放不下）",
        mut(_break_down_ctx))

    def _break_paged(m):
        for c in m["matrix.json"]["cells"]:
            if c["feasible"] and c["backend"] == "vllm-paged" and c.get("memory"):
                c["memory"]["kv_bytes"] = c["memory"]["peak_bytes"]
                return
    add("paged_backends_do_not_reserve_the_whole_context",
        "把分页后端的 KV 占用抬到峰值（= 分页退化成连续预留）",
        mut(_break_paged))

    def _break_nogpu(m):
        for c in m["matrix.json"]["cells"]:
            if c["hardware"] == "epyc-cpu-64g" and c["backend"] == "vllm-paged":
                c["feasible"] = True
                c["reason"] = "ok"
                return
    add("no_gpu_hardware_rejects_gpu_backends",
        "让纯 CPU 机器上的 GPU 后端变成可行（= 没有 GPU 也能跑 GPU 内核）",
        mut(_break_nogpu))

    def _break_qsup(m):
        for c in m["matrix.json"]["cells"]:
            if c["reason"] == "quant_unsupported":
                c["reason"] = "ok"
                c["feasible"] = True
                c["readings"] = {}
                return
    add("unsupported_quant_is_marked_not_silently_computed",
        "把「量化不支持」改成「可行但没读数」（= 静默算了一个跑不了的组合）",
        mut(_break_qsup))

    def _break_oom(m):
        for c in m["matrix.json"]["cells"]:
            if c["reason"] == "gpu_oom":
                c["reason"] = "ram_oom"
                return
    add("oom_cells_really_exceed_the_device_budget",
        "把一条 gpu_oom 的原因改成 ram_oom（= 归因与实际超出的预算不符）",
        mut(_break_oom))

    def _break_fit(m):
        for c in m["matrix.json"]["cells"]:
            if c["feasible"] and c.get("memory"):
                c["memory"]["peak_bytes"] = 10 ** 18
                return
    add("feasible_cells_fit_the_device_budget",
        "把某个可行格的峰值占用抬到 1 EB（= 可行域没看容量）",
        mut(_break_fit))

    # ---- 报告 / 渲染器 ----
    def _break_pure(m):
        m["summary.json"] = copy.deepcopy(m.get("summary.json") or {})
        m["summary.json"].setdefault("n_cells", 0)
    add("renderers_are_pure_functions_of_the_products",
        "往产物里塞一个渲染器不认识的键（= 报告的输入超出了产物集）",
        mut(_break_pure))

    def _break_artifact_checksum(m):
        a = m["artifact.json"]
        if isinstance(a, list) and a:
            a[0]["checksum"] = "deadbeef" * 8
        elif isinstance(a, dict) and a.get("artifacts"):
            a["artifacts"][0]["checksum"] = "deadbeef" * 8
        else:
            a["checksum"] = "deadbeef" * 8
    add("artifact_checksum_matches_its_own_body",
        "把 Artifact 的校验和改成错的（= 清单与内容对不上）",
        mut(_break_artifact_checksum))

    def _break_deploy(m):
        d = m["deploy.json"]
        rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
        if isinstance(rows, list) and rows:
            rows[0]["meets_slo"] = not rows[0].get("meets_slo", True)
    add("deploy_rows_recheck_the_slo_against_the_artifact",
        "把部署行的 SLO 判定翻面（= 部署清单与它引用的格子不一致）",
        mut(_break_deploy))

    # ---- 时间键：塞一个**实测**耗时 ----
    def _inject_elapsed(m):
        m["matrix.json"]["cells"][0]["readings"]["wall_clock_s"] = 1.23
    add("no_undeclared_time_key_in_the_versioned_products",
        "往读数里塞一个 `wall_clock_s`（= 实测耗时混进模型预测，"
        "这正是这个判据存在的理由）",
        mut(_inject_elapsed))

    # ---- 输入完整性 ----
    def _break_inputs(m):
        m.pop("matrix.json", None)
    add("adversarial_cases_all_ran_with_their_inputs_present",
        "删掉 matrix.json（= 对抗层该读的产物不在位）",
        mut(_break_inputs))

    return T


# --------------------------------------------------------------------- 主流程
def main() -> int:
    cfg = load_config(ROOT / "configs" / "default.json")
    products = {}
    for f in sorted((ROOT / "out").glob("*.json")):
        products[f.name] = json.loads(f.read_text(encoding="utf-8"))

    ms = tuple(SP.load_models(cfg))
    hs = tuple(SP.load_hardware(cfg))

    # ---- 基线：每条必须先是绿的 ----------------------------------------
    base = INV.run_adversarial(products, cfg, ms, hs)
    red = [r["name"] for r in base["cases"] if not r["held"]]
    if red:
        print("❌ 基线就有红的用例，先把它们修绿：")
        for r in base["cases"]:
            if not r["held"]:
                print(f"   - {r['name']}: {r['detail'][:110]}")
        return 1
    print(f"✅ 基线 {base['n_held']}/{base['n_cases']} 全绿，开始逐条变异")

    rng = random.Random(SEED)
    specs = {s["name"]: s for s in INV._CASES}
    targeted = targeted_mutations(products, cfg, ms, hs)

    results = []
    n_generic = n_target = n_unproven = 0
    for name in sorted(specs):
        fn = specs[name]["fn"]

        # ---- 探针：记下这条用例真正读过的键 ----------------------------
        probe = Probe(products)
        try:
            fn(probe, cfg, ms, hs)
        except Exception:                                # noqa: BLE001
            pass
        touched = set(probe.touched)

        killer = None
        kind = None

        # ---- ① 通用变异（只动产物）------------------------------------
        for desc, mutated in generic_mutations(products, touched, rng):
            try:
                held, _ = fn(mutated, cfg, ms, hs)
            except Exception:                            # noqa: BLE001
                held = False
            if not held:
                killer, kind = desc, "generic"
                break

        # ---- ② 变异**配置对象** ----------------------------------------
        # ⚠️ 一批用例读 `c` 不读 `p`（轴、目录、指纹）——
        #    只动产物的话它们永远打不动，会被误判成恒真。
        if killer is None:
            for desc, alt in cfg_mutations(cfg, touched):
                try:
                    held, _ = fn(products, alt, ms, hs)
                except Exception:                        # noqa: BLE001
                    held = False
                if not held:
                    killer, kind = desc, "cfg"
                    break

        # ---- ③ 打源码补丁（纯函数）------------------------------------
        if killer is None:
            for why, gen in src_patches():
                g = gen()
                try:
                    next(g)
                    try:
                        held, _ = fn(products, cfg, ms, hs)
                    except Exception:                    # noqa: BLE001
                        held = False
                finally:
                    try:
                        next(g)
                    except StopIteration:
                        pass
                if not held:
                    killer, kind = why, "source"
                    break

        # ---- ④ 定向变异 -----------------------------------------------
        if killer is None:
            for why, make in targeted.get(name, []):
                mutated = make()
                try:
                    held, _ = fn(mutated, cfg, ms, hs)
                except Exception:                        # noqa: BLE001
                    held = False
                if not held:
                    killer, kind = f"{why}", "targeted"
                    break

        if killer is None:
            n_unproven += 1
            print(f"  ❓ {name:56s} 通用与定向变异都没能打红")
        elif kind == "generic":
            n_generic += 1
        else:
            n_target += 1
        results.append({"name": name, "killer": killer, "kind": kind})

    print()
    print("=" * 78)
    unproven = [r for r in results if r["killer"] is None]
    if unproven:
        print(f"❌ {len(unproven)} 条用例**没能被证明能红**：")
        for r in unproven:
            print(f"   - {r['name']}")
        print("   ⇒ 它们可能恒真，也可能只是变异没打中靶 —— 必须分清。")
    else:
        print(f"✅ 全部 {len(results)} 条用例都能红"
              f"（通用变异 {n_generic} 条 / 定向变异 {n_target} 条）")

    (ROOT / "diagnostics" / "mutation.json").write_text(
        json.dumps({"n_cases": len(results),
                    "n_proven": len(results) - len(unproven),
                    "n_generic": n_generic, "n_targeted": n_target,
                    "seed": SEED, "max_mut": MAX_MUT,
                    "results": results}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8", newline="\n")
    print("结论已落 diagnostics/mutation.json")
    return 0 if not unproven else 1


if __name__ == "__main__":
    raise SystemExit(main())
