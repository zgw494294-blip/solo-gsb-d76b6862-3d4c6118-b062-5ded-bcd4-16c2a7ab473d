"""测站基准突变定位：对潮汐拟合残差做带约束的 L1 分段（中位数偏移）。

把逐点残差划分为若干连续段，每段以段内残差的**较小中位数**
（偶数长度时取两个中位数中偏小者，即排序后下标 (m-1)//2 的元素）
作为该段的基准偏移 μ，最小化

    F = Σ_i |r_i - μ_{seg(i)}| + penalty × k

其中 k 为变点数（段数 - 1），每个段长度不得少于 min_segment。

到中位数的绝对差之和有闭式速算：排序后长 m 的段，代价等于最大的
floor(m/2) 个值之和减去最小的 floor(m/2) 个值之和（奇、偶长度皆然，
偶数长度时偏移取较小中位数，代价不变）。段代价始终由同一条
``np.sort`` 算术路径算出（排序后上下两半各自求和再相减），因此
DP 比较、返回的 total_cost 与按定义重算的绝对差之和位级一致，
并列裁决不依赖浮点求和顺序。

动态规划：dp_s(j) = 前 j 个点恰好分 s 段（每段 ≥ L）的最小目标值，
每新增一段（即新增一个变点）计入一次 penalty。最优方案并列时，
先取变点少者，再取变点索引序列字典序最小者——因此每个状态除目标
值外还保留完整的变点索引元组，按 (目标值, 变点数, 索引元组) 比较。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ShiftResult:
    """基准突变定位结果。

    boundaries: 内部变点索引（新段首点下标），严格递增
    starts/ends: 各段 [start, end) 的起止下标，len = 段数
    offsets: 各段偏移（段内残差的较小中位数）
    corrected: 校正后逐点残差 r_i - μ_{seg(i)}
    total_cost: 最优目标值（绝对差之和 + penalty × 变点数）
    """

    boundaries: tuple[int, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    offsets: tuple[float, ...]
    corrected: np.ndarray
    total_cost: float


def segment_cost(segment: np.ndarray) -> float:
    """单个段到其较小中位数的绝对差之和（唯一规范算术路径）。

    排序后长 m、h = m//2：上 h 个之和减去下 h 之和。奇、偶长度皆成立
    （奇数时长 2h+1，中位元素不参与；偶数时偏移取较小中位数
    ``ordered[h-1]``，到它的绝对差之和同样等于该差值）。
    """
    ordered = np.sort(segment)
    half = ordered.size // 2
    if half == 0:
        return 0.0
    return float(ordered[-half:].sum() - ordered[:half].sum())


def precompute_segment_costs(
    residuals: np.ndarray, min_segment: int
) -> list[list[float]]:
    """计算所有长度 ≥ min_segment 的段 [i, j) 的绝对差代价。

    每个可行段独立排序后按上下两半之和相减（见 segment_cost）；
    不可行的 (i, j) 记 +inf。n ≤ 500 时段数至多约 12.5 万，直接枚举。
    """
    n = len(residuals)
    r = np.asarray(residuals, dtype=np.float64)
    inf = float("inf")
    cost: list[list[float]] = [[inf] * (n + 1) for _ in range(n)]

    for start in range(n - min_segment + 1):
        for end in range(start + min_segment, n + 1):
            cost[start][end] = segment_cost(r[start:end])
    return cost


def solve_segmentation(
    cost: list[list[float]],
    n: int,
    min_segment: int,
    max_segments: int,
    penalty: float,
) -> tuple[tuple[int, ...], float]:
    """在段长约束下求最优分段，返回 (内部变点索引元组, 最优目标值)。

    cost[i][j] 为段 [i, j) 的绝对差代价；max_segments 为段数上限。
    """
    length = min_segment
    max_segments = min(max_segments, n // length)

    # 1 段：无前驱变点，目标值即整段代价。
    prev: dict[int, tuple[float, tuple[int, ...]]] = {
        end: (cost[0][end], ()) for end in range(length, n + 1)
    }
    best_obj, best_changes, best_tuple = prev[n][0], 0, ()

    for segments in range(2, max_segments + 1):
        cur: dict[int, tuple[float, tuple[int, ...]]] = {}
        for end in range(segments * length, n + 1):
            winner: tuple[float, tuple[int, ...]] | None = None
            # 末段为 [mid, end)，前驱覆盖 [0, mid) 且共 segments-1 段。
            for mid in range((segments - 1) * length, end - length + 1):
                prefix = prev.get(mid)
                if prefix is None:
                    continue
                candidate = (
                    prefix[0] + cost[mid][end] + penalty,
                    prefix[1] + (mid,),
                )
                if winner is None or candidate < winner:
                    winner = candidate
            if winner is not None:
                cur[end] = winner

        candidate = cur.get(n)
        if candidate is not None:
            # 并列：先比变点少，再比索引元组字典序。
            candidate_key = (candidate[0], segments - 1, candidate[1])
            best_key = (best_obj, best_changes, best_tuple)
            if candidate_key < best_key:
                best_obj, best_changes, best_tuple = candidate_key
        prev = cur

    return best_tuple, float(best_obj)


def detect_shifts(
    residuals: np.ndarray,
    min_segment: int,
    max_changes: int,
    penalty: float,
) -> ShiftResult:
    """定位残差序列中的持续基准偏移。"""
    r = np.asarray(residuals, dtype=np.float64)
    n = r.size

    cost = precompute_segment_costs(r, min_segment)
    boundaries, total_cost = solve_segmentation(
        cost, n, min_segment, max_changes + 1, float(penalty)
    )

    starts = (0,) + boundaries
    ends = boundaries + (n,)

    offsets: list[float] = []
    corrected = np.empty(n, dtype=np.float64)
    for start, end in zip(starts, ends):
        segment = r[start:end]
        offset = float(np.sort(segment)[(end - start - 1) // 2])
        offsets.append(offset)
        corrected[start:end] = segment - offset

    return ShiftResult(
        boundaries=boundaries,
        starts=starts,
        ends=ends,
        offsets=tuple(offsets),
        corrected=corrected,
        total_cost=total_cost,
    )
