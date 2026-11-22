"""确定性随机源。

⚠️ **不用 `random`**：它的 Mersenne Twister 状态是进程级的，任何一处
偷偷调一次 `random.random()` 就会让「两次 run 逐位相同」失效，而症状是
「产物偶尔不同」—— 最难查的那类。

⚠️ **不用 `hash()`**：字符串哈希在 Python 3 里是**加盐**的（PYTHONHASHSEED），
同一段代码两次进程的 `hash("x")` 不同。
"""

from __future__ import annotations

import math

_MASK64 = (1 << 64) - 1
#: 53 位均匀量的最小正值 `2⁻⁵³`（`uniform()` 返回 0 时用它兜底）。
_MIN_UNIFORM = 1.0 / 9007199254740992.0
#: LCG 参数（Numerical Recipes 的 `MMIX` 变体：乘数与增量都是公开的固定常数）
_A = 6364136223846793005
_C = 1442695040888963407


class Rng:
    """64 位 LCG。逐位可复现，与 Python 版本无关。"""

    __slots__ = ("_s",)

    def __init__(self, seed: int) -> None:
        self._s = (seed * _A + _C) & _MASK64

    def next_u64(self) -> int:
        self._s = (self._s * _A + _C) & _MASK64
        return self._s

    def uniform(self) -> float:
        """`[0, 1)` 上的均匀量（53 位精度）。

        ⚠️ 值域是**左闭**的：`u == 0.0` 可能出现（概率 `2⁻⁵³`）。
        依赖 `log(u)` 的调用方必须自己兜底（`exponential` 把 0 抬到
        `2⁻⁵³`）—— 否则 `math.log(0.0)` 抛 `ValueError`，而这条路径
        极难复现，正是「跑一万次都没事，线上偶尔炸」的那一类。
        """
        return (self.next_u64() >> 11) / 9007199254740992.0  # 2**53

    def exponential(self, rate: float) -> float:
        """`Exp(rate)` 采样（逆变换：`-ln(U)/rate`）。"""
        u = self.uniform()
        if u <= 0.0:
            u = _MIN_UNIFORM
        return -math.log(u) / rate

    def exponentials(self, n: int, rate: float) -> list[float]:
        """批量采样 `n` 个 `Exp(rate)`，序列与逐个调用**逐位相同**。

        ⚠️ 只为热路径：互校的 54 个模拟点 × 4000 请求 ⇒ 43 万次
        `exponential` 调用，每次都要做 `uniform` / `next_u64` 的属性查找。
        这里的循环保持**完全相同的**随机数取用顺序（先 `n` 个到达间隔、
        再 `n` 个服务时长），所以产物逐位不变。
        """
        out: list[float] = []
        append = out.append
        next_u64 = self.next_u64
        log = math.log
        for _ in range(n):
            u = (next_u64() >> 11) / 9007199254740992.0  # 2**53
            if u <= 0.0:
                u = _MIN_UNIFORM
            append(-log(u) / rate)
        return out

    def fork(self, tag: str) -> "Rng":
        """从 `tag` 派生一个独立流。

        ⚠️ 派生必须**确定性**：`hash(tag)` 在跨进程不确定，
        所以这里逐字符累加（`_A` 是常数）。
        """
        acc = self._s
        for ch in tag.encode("utf-8"):
            acc = (acc * _A + _C + ch) & _MASK64
        return Rng(acc & _MASK64)


def stream_for(*parts: str, seed: int) -> Rng:
    """从一批标签派生一条流（顺序敏感、跨进程稳定）。"""
    r = Rng(seed)
    for p in parts:
        r = r.fork(p)
    return r
