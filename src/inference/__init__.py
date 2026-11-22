"""InferenceFactory：Model → Backend → Benchmark → 部署。

形状：**扇出-收拢型第四形态（部署配置空间）**。
候选从 `model × hardware × backend × quant × batch × context × arrival` 的笛卡尔积
展开（**全枚举，不是搜索** ⇒ 验收第一条是「矩阵缺口 = 0」），逐点测量一次
`InferenceProfile`，收拢成一个**在 VRAM 硬约束 + P99 SLO 下可行的部署决策**。

与同族的 `L5-quantfactory`（第三形态：量化设计空间）多的两样：
① 空间里有一维是**随机过程**（到达率 ⇒ M/M/c 排队 ⇒ P99 是分位数不是确定性读数）；
② 硬约束是**内生的** —— VRAM 占用是配置（batch × context）的函数，
   所以「可行域」本身是要算出来的产物，不是外部给的阶梯。
"""

__all__ = ["__version__"]
__version__ = "1.0.0"
