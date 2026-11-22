"""确定性随机源。

⚠️ **不用 `random`**：它的 Mersenne Twister 状态是进程级的，任何一处
偷偷调一次 `random.random()` 就会让「两次 run 逐位相同」失效，而症状是
「产物偶尔不同」—— 最难查的那类。

⚠️ **不用 `hash()`**：字符串哈希在 Python 3 里是**加盐**的（PYTHONHASHSEED），
同一段代码两次进程的 `hash("x")` 不同。
"""

from __future__ import annotations

_MASK64 = (1 << 64) - 1
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
        """(0, 1) 上的均匀量。

        ⚠️ 用 `u64 + 0.5` 而不是 `u64 / 2**64`：后者可能返回 0.0，
        而 `log(0)` 是 `-inf` ⇒ 指数分布采样会给出 `inf` 的服务时间。
        """
        return (self.next_u64() >> 11) / 9007199254740992.0  # 2**53

    def exponential(self, rate: float) -> float:
        """`Exp(rate)` 采样。"""
        import math

        return -math.log(self.uniform()) / rate

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
