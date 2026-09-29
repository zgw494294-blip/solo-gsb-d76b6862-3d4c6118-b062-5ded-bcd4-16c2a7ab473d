"""测站基准突变定位：残差分段常数（中位数）L1 最优分割。

把 /v1/tide/fit 返回的逐点残差视为「分段常数偏移 + 噪声」，在
最多少数几个变点、且每段长度不少于 ``min_segment`` 的约束下，寻找

    目标 = Σ_i | r_i - o_{seg(i)} | + penalty × 变点数

最小的分割方案；每段偏移 o 取该段残差的**较小中位数**（升序后下标
``(L-1)//2`` 的元素）。中位数使绝对偏差之和最小，对偶长段存在两个
最优中位数，按题意固定取较小者，方案因此唯一可复现。

并列裁决（总目标相同的可行方案之间）：

1. 变点少者优先（即段数少者优先）；
2. 变点索引序列字典序小者优先（等价于全段边界 0, c1, ..., n 的
   字典序在同段数前提下的比较结果）。

实现：

* 代价表 cost[i][j]（i<j，段为残差下标 [i, j)）用「双堆 + 两侧和」
  增量维护：大根堆 low 存较小的一半、小根堆 high 存较大的一半，
  low 堆顶即较小中位数；两侧和随插入同步累加，直接得到绝对偏差
  之和，无需对每个区间排序。
* 分割用动态规划：dp[s][j] = 前 j 点恰好分 s 段的最小偏差和与
  对应边界元组；边界元组按字典序比较，天然实现规则 2。
"""
from __future__ import annotations

import heapq
from typing import Sequence


class _MaxHeap:
    """以存负值的 list 承载的大根堆。"""

    __slots__ = ("_h",)

    def __init__(self) -> None:
        self._h: list[float] = []

    def push(self, x: float) -> None:
        heapq.heappush(self._h, -x)

    def pop(self) -> float:
        return -heapq.heappop(self._h)

    def peek(self) -> float:
        return -self._h[0]

    def __len__(self) -> int:
        return len(self._h)


def build_segment_tables(
    residuals: Sequence[float],
) -> tuple[list[list[float]], list[list[float]]]:
    """预计算所有区间 [i, j)（i<j）的较小中位数与 L1 代价。

    返回 (medians, costs)，均为 (n+1)×(n+1) 表，仅上三角 (i, j>i) 有效。
    """
    n = len(residuals)
    medians: list[list[float]] = [[0.0] * (n + 1) for _ in range(n + 1)]
    costs: list[list[float]] = [[0.0] * (n + 1) for _ in range(n + 1)]

    for i in range(n):
        low = _MaxHeap()  # 较小的一半；堆顶为较小中位数
        high: list[float] = []  # 较大的一半（小根堆）
        sum_low = 0.0
        sum_high = 0.0

        for j in range(i + 1, n + 1):
            x = float(residuals[j - 1])
            if not low or x <= low.peek():
                low.push(x)
                sum_low += x
            else:
                heapq.heappush(high, x)
                sum_high += x

            # 不变量：len(low) == len(high)（偶长）
            #    或 len(low) == len(high) + 1（奇长）。
            # 每次插入至多失衡 1，用 if 即可恢复。
            if len(low) > len(high) + 1:
                moved = low.pop()
                sum_low -= moved
                heapq.heappush(high, moved)
                sum_high += moved
            elif len(low) < len(high):
                moved = heapq.heappop(high)
                sum_high -= moved
                low.push(moved)
                sum_low += moved

            med = low.peek()
            # Σ|r - med| = (med·|low| - Σlow) + (Σhigh - med·|high|)
            cost = med * len(low) - sum_low + sum_high - med * len(high)

            medians[i][j] = med
            costs[i][j] = cost

    return medians, costs


def locate_shifts(
    residuals: Sequence[float],
    times: Sequence[int],
    min_segment: int,
    max_changes: int,
    penalty: float,
) -> dict:
    """求基准突变的最优分段。

    返回字段：

    * segments: 每段 {"start", "end", "offset"}，区间为 [start, end)
    * offsets: 各段偏移（与 segments 同序）
    * change_points: 每个变点 {"index", "time"}，index/time 为该变点
      所开启新段的首点下标与对应整数分钟时刻
    * corrected_residuals: 逐点 r_i - o_seg(i)
    * objective: 总目标值（偏差和 + penalty × 变点数）
    """
    n = len(residuals)
    medians, costs = build_segment_tables(residuals)

    # 第 1 层：前 j 点为一个段 [0, j)，所有 j ≥ min_segment 均为可行状态
    # （后续层需要把任意 k 处的单段前缀作为递推起点）。
    prev_dp: list[tuple[float, tuple[int, ...]] | None] = [None] * (n + 1)
    prev_dp[0] = (0.0, (0,))
    for j in range(min_segment, n + 1):
        prev_dp[j] = (costs[0][j], (0, j))

    # s = 1 的整段方案（schema 已保证 1 ≤ min_segment ≤ n，故恒可行）。
    best_overall: tuple[float, int, tuple[int, ...]] = (costs[0][n], 1, (0, n))

    max_segments = min(max_changes + 1, n // min_segment)
    for s in range(2, max_segments + 1):
        cur_dp: list[tuple[float, tuple[int, ...]] | None] = [None] * (n + 1)
        # 前 j 点分 s 段：最后一段 [k, j) 长度 ≥ min_segment，
        # 前 k 点分 s-1 段，故 k ≥ (s-1)·min_segment、k ≤ j-min_segment。
        k_min = (s - 1) * min_segment
        for j in range(s * min_segment, n + 1):
            best: tuple[float, tuple[int, ...]] | None = None
            for k in range(k_min, j - min_segment + 1):
                candidate_prev = prev_dp[k]
                if candidate_prev is None:
                    continue
                value = candidate_prev[0] + costs[k][j]
                candidate = (value, candidate_prev[1] + (j,))
                if best is None or candidate < best:
                    best = candidate
            cur_dp[j] = best

        current = cur_dp[n]
        if current is not None:
            value, boundaries = current
            objective = value + (s - 1) * penalty
            # 全局键：总目标 → 段数（少者优先）→ 边界元组字典序。
            candidate_key = (objective, s, boundaries)
            best_key = (best_overall[0], best_overall[1], best_overall[2])
            if candidate_key < best_key:
                best_overall = (objective, s, boundaries)
        prev_dp = cur_dp

    objective, s_count, boundaries = best_overall

    segments: list[dict] = []
    offsets: list[float] = []
    corrected = [0.0] * n
    for seg_idx in range(s_count):
        start, end = boundaries[seg_idx], boundaries[seg_idx + 1]
        offset = medians[start][end]
        offsets.append(offset)
        segments.append({"start": start, "end": end, "offset": offset})
        for point in range(start, end):
            corrected[point] = float(residuals[point]) - offset

    change_points = [
        {"index": boundaries[q], "time": int(times[boundaries[q]])}
        for q in range(1, s_count)
    ]

    return {
        "segments": segments,
        "offsets": offsets,
        "change_points": change_points,
        "corrected_residuals": corrected,
        "objective": float(objective),
    }
