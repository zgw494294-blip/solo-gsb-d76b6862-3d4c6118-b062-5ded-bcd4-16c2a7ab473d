"""POST /v1/tide/datum-shift 的端到端测试。

覆盖：阶跃偏移定位、penalty=0 并列时的「变点少优先 / 字典序最小」
裁决、段长约束、max_changes=0、422 校验、与暴力枚举最优值对拍，
以及与 /v1/tide/fit 串接的完整流程。
"""
from __future__ import annotations

import itertools
import json

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.segments import build_segment_tables

client = TestClient(app)

PATH = "/v1/tide/datum-shift"


def post_raw(payload: dict) -> "TestClient":  # type: ignore[name-defined]
    """以非严格 JSON 发送（允许 NaN、Infinity token 到达服务端）。"""
    return client.post(
        PATH,
        content=json.dumps(payload, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )


def lower_median(values: list[float]) -> float:
    return sorted(values)[(len(values) - 1) // 2]


def test_segment_tables_match_naive_sorting():
    """双堆增量代价表必须与「排序取较小中位数」的朴素算法完全一致。"""
    rng = np.random.default_rng(55)
    residuals = rng.normal(0.0, 1.0, 40).tolist()
    medians, costs = build_segment_tables(residuals)
    n = len(residuals)
    for i in range(n):
        for j in range(i + 1, n + 1):
            window = residuals[i:j]
            med = lower_median(window)
            assert medians[i][j] == med, (i, j)
            assert abs(costs[i][j] - sum(abs(x - med) for x in window)) < 1e-10


def test_segment_tables_lower_median_on_even_length():
    """偶长段必须取下中位数（两中位数中的较小者）。"""
    medians, costs = build_segment_tables([1.0, 2.0, 3.0, 100.0])
    assert medians[0][4] == 2.0  # 下中位数为 2.0 而非 3.0
    assert costs[0][4] == abs(1 - 2) + 0 + abs(3 - 2) + abs(100 - 2)
    assert medians[0][2] == 1.0
    assert costs[0][2] == 1.0


def brute_force_best(
    residuals: list[float],
    min_segment: int,
    max_changes: int,
    penalty: float,
) -> tuple[float, int, tuple[int, ...]]:
    """枚举所有满足段长约束的分割，返回 (总目标, 段数, 边界元组) 最优键。"""
    n = len(residuals)
    best = None
    for s in range(1, min(max_changes + 1, n // min_segment) + 1):
        # 在 1..n-1 中选 s-1 个内部边界，且相邻边界（含 0、n）间距 ≥ min_segment。
        for inner in itertools.combinations(range(1, n), s - 1):
            bounds = (0,) + inner + (n,)
            if any(b - a < min_segment for a, b in zip(bounds, bounds[1:])):
                continue
            total = 0.0
            for a, b in zip(bounds, bounds[1:]):
                med = lower_median(residuals[a:b])
                total += sum(abs(x - med) for x in residuals[a:b])
            total += penalty * (s - 1)
            key = (total, s, bounds)
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def test_detects_single_step_shift():
    """干净的单阶跃：应在阶跃处给出唯一变点，目标为 0。"""
    n = 40
    t = (np.arange(n) * 30).tolist()
    residuals = [0.0] * 20 + [3.5] * 20
    resp = client.post(
        PATH,
        json={
            "t": t,
            "residuals": residuals,
            "min_segment": 5,
            "max_changes": 4,
            "penalty": 0.0,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert len(data["segments"]) == 2
    assert data["segments"][0] == {"start": 0, "end": 20, "offset": 0.0}
    assert data["segments"][1] == {"start": 20, "end": 40, "offset": 3.5}
    assert data["offsets"] == [0.0, 3.5]
    assert data["change_points"] == [{"index": 20, "time": t[20]}]
    assert data["corrected_residuals"] == [0.0] * n
    assert data["objective"] == 0.0


def test_penalty_zero_prefers_fewer_changes_on_ties():
    """常值块内可零代价任意切分；并列时必须选择变点最少的方案。"""
    n = 40
    t = list(range(n))
    residuals = [-1.0] * 20 + [1.0] * 20
    resp = client.post(
        PATH,
        json={
            "t": t,
            "residuals": residuals,
            "min_segment": 5,
            "max_changes": 4,
            "penalty": 0.0,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["segments"]) == 2  # 而非 3/4/5 段的零代价碎切
    assert [cp["index"] for cp in data["change_points"]] == [20]
    assert data["objective"] == 0.0


def test_penalty_zero_lexicographically_smallest_boundaries():
    """多个切点总代价并列时，取边界字典序最小者。"""
    n = 30
    # 三段等距常值 0/1/2 各 10 点；限制最多 1 个变点（两段）时，
    # 切点 k=10..20 的总绝对偏差都为 10（切点落在中间常值块内时，
    # 中位数把两侧的 0/2 代价均摊为常数），应选字典序最小的 k=10。
    residuals = [0.0] * 10 + [1.0] * 10 + [2.0] * 10
    resp = client.post(
        PATH,
        json={
            "t": list(range(n)),
            "residuals": residuals,
            "min_segment": 5,
            "max_changes": 1,
            "penalty": 0.0,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["segments"]) == 2
    assert [cp["index"] for cp in data["change_points"]] == [10]
    assert data["segments"][0]["offset"] == 0.0
    assert data["segments"][1]["offset"] == 1.0
    assert abs(data["objective"] - 10.0) < 1e-12


def test_penalty_discourages_extra_changes():
    """penalty 足够大时，即便真实存在小阶跃也应退回单段。"""
    n = 30
    t = list(range(n))
    residuals = [0.0] * 15 + [0.2] * 15
    high_penalty = client.post(
        PATH,
        json={
            "t": t,
            "residuals": residuals,
            "min_segment": 5,
            "max_changes": 2,
            "penalty": 100.0,
        },
    ).json()
    assert len(high_penalty["segments"]) == 1
    assert high_penalty["change_points"] == []
    seg = high_penalty["segments"][0]
    assert seg["start"] == 0 and seg["end"] == n
    # 单段偏移取较小中位数：30 个点中前 15 个 0.0 → 下标 14 为 0.0
    assert seg["offset"] == 0.0
    assert abs(high_penalty["objective"] - 15 * 0.2) < 1e-12

    zero_penalty = client.post(
        PATH,
        json={
            "t": t,
            "residuals": residuals,
            "min_segment": 5,
            "max_changes": 2,
            "penalty": 0.0,
        },
    ).json()
    assert len(zero_penalty["segments"]) == 2
    assert [cp["index"] for cp in zero_penalty["change_points"]] == [15]
    assert zero_penalty["objective"] == 0.0


def test_max_changes_zero_forces_single_segment():
    n = 30
    t = list(range(n))
    residuals = [0.0] * 10 + [5.0] * 10 + [-5.0] * 10
    data = client.post(
        PATH,
        json={
            "t": t,
            "residuals": residuals,
            "min_segment": 5,
            "max_changes": 0,
            "penalty": 0.0,
        },
    ).json()
    assert len(data["segments"]) == 1
    assert data["change_points"] == []
    assert data["segments"][0] == {"start": 0, "end": 30, "offset": 0.0}
    assert abs(data["objective"] - 100.0) < 1e-12


def test_min_segment_constraint_enforced():
    """真实切点距边界不足 min_segment 时不允许使用，须退而求其次。"""
    n = 30
    t = list(range(n))
    residuals = [0.0] * 3 + [4.0] * 27  # 真正分界在 3，但 min_segment=5
    data = client.post(
        PATH,
        json={
            "t": t,
            "residuals": residuals,
            "min_segment": 5,
            "max_changes": 4,
            "penalty": 0.0,
        },
    ).json()
    # 唯一可行（且字典序最小）的切点是 5；此后 25 个常值点内的零代价
    # 碎切因「变点少优先」不会出现。
    assert [cp["index"] for cp in data["change_points"]] == [5]
    seg0, seg1 = data["segments"]
    assert (seg0["start"], seg0["end"]) == (0, 5)
    assert (seg1["start"], seg1["end"]) == (5, 30)
    # [0,5) = [0,0,0,4,4] 的较小中位数为 0
    assert seg0["offset"] == 0.0
    assert seg1["offset"] == 4.0
    assert abs(data["objective"] - 8.0) < 1e-12  # 前 5 点贡献 2×4


def test_corrected_residuals_and_objective_consistency():
    rng = np.random.default_rng(7)
    n = 50
    t = (np.arange(n) * 10).tolist()
    residuals = (
        np.concatenate(
            [rng.normal(0.0, 0.05, 20), rng.normal(1.5, 0.05, 30)]
        ).tolist()
    )
    data = client.post(
        PATH,
        json={
            "t": t,
            "residuals": residuals,
            "min_segment": 8,
            "max_changes": 3,
            "penalty": 1.25,
        },
    ).json()

    # 段区间覆盖 [0, n)、互不重叠且满足段长约束
    bounds = [(s["start"], s["end"]) for s in data["segments"]]
    assert bounds[0][0] == 0 and bounds[-1][1] == n
    assert all(b[1] - b[0] >= 8 for b in bounds)
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))

    # 变点时刻取自 t，偏移与段一致
    for cp, seg in zip(data["change_points"], data["segments"][1:]):
        assert cp["index"] == seg["start"]
        assert cp["time"] == t[cp["index"]]
    assert [s["offset"] for s in data["segments"]] == data["offsets"]

    # 校正残差与目标值可以由分段结果直接复算
    recomputed = []
    abs_sum = 0.0
    for seg in data["segments"]:
        for i in range(seg["start"], seg["end"]):
            corrected = residuals[i] - seg["offset"]
            recomputed.append(corrected)
            abs_sum += abs(corrected)
    assert np.allclose(data["corrected_residuals"], recomputed, atol=1e-12)
    expected_objective = abs_sum + 1.25 * len(data["change_points"])
    assert abs(data["objective"] - expected_objective) < 1e-9


def test_matches_brute_force_on_random_instances():
    """随机小实例上，接口结果的目标值/段数/边界必须与暴力枚举完全一致。"""
    rng = np.random.default_rng(123)
    for trial in range(12):
        n = int(rng.integers(20, 31))
        min_segment = int(rng.integers(5, 9))
        max_changes = int(rng.integers(0, 3))
        penalty = float(rng.choice([0.0, 0.1, 1.0, 5.0]))

        # 以分段常数 + 噪声生成，保证存在若干真实变点
        residuals = rng.normal(0.0, 0.05, n).tolist()
        for cp in sorted(rng.choice(range(min_segment, n - min_segment),
                                    size=max_changes, replace=False)):
            shift = float(rng.choice([-2.0, 2.0]))
            for i in range(cp, n):
                residuals[i] += shift

        resp = client.post(
            PATH,
            json={
                "t": list(range(n)),
                "residuals": residuals,
                "min_segment": min_segment,
                "max_changes": max_changes,
                "penalty": penalty,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()

        best_obj, best_s, best_bounds = brute_force_best(
            residuals, min_segment, max_changes, penalty
        )
        assert abs(data["objective"] - best_obj) < 1e-9, (trial, data["objective"], best_obj)
        assert len(data["segments"]) == best_s
        assert [s["start"] for s in data["segments"]] == list(best_bounds[:-1])
        assert [s["end"] for s in data["segments"]] == list(best_bounds[1:])
        for seg, (a, b) in zip(data["segments"], zip(best_bounds[:-1], best_bounds[1:])):
            assert abs(seg["offset"] - lower_median(residuals[a:b])) < 1e-12


def test_chained_with_tide_fit_recovers_injected_shift():
    """端到端串接：fit 的 residuals 直接送入 datum-shift，定位注入的基准突变。"""
    n = 60
    step = 30
    tt = np.arange(n) * step
    rng = np.random.default_rng(99)
    h = (
        2.5
        + 1.2 * np.cos(2 * np.pi * tt / 745.0)
        - 0.7 * np.sin(2 * np.pi * tt / 745.0)
        + rng.normal(0.0, 0.02, n)
    )
    h[35:] += 1.0  # 第 35 个点起基准整体抬升 1 m
    t = tt.tolist()

    fit = client.post(
        "/v1/tide/fit",
        json={"t": t, "h": h.tolist(), "P": [745], "delta": 0.2},
    )
    assert fit.status_code == 200, fit.text
    residuals = fit.json()["residuals"]

    shifted = client.post(
        PATH,
        json={
            "t": t,
            "residuals": residuals,
            "min_segment": 10,
            "max_changes": 2,
            "penalty": 1.0,
        },
    )
    assert shifted.status_code == 200, shifted.text
    data = shifted.json()
    assert len(data["change_points"]) == 1
    cp = data["change_points"][0]
    assert cp["index"] == 35
    assert cp["time"] == t[35]
    offsets = data["offsets"]
    assert abs(offsets[0] - 0.0) < 0.2
    assert abs(offsets[1] - 1.0) < 0.2
    # 校正后的残差应全部回到噪声/拟合失配量级
    assert max(abs(r) for r in data["corrected_residuals"]) < 0.25


def test_validation_errors_are_422():
    n = 25
    base = {
        "t": list(range(n)),
        "residuals": [0.0] * n,
        "min_segment": 5,
        "max_changes": 2,
        "penalty": 1.0,
    }

    def post(**overrides):
        return client.post(PATH, json={**base, **overrides})

    assert post(t=list(range(19)), residuals=[0.0] * 19).status_code == 422  # 点数 < 20
    assert post(t=list(range(501)), residuals=[0.0] * 501).status_code == 422  # > 500
    bad_t = list(range(n))
    bad_t[10] = bad_t[9]
    assert post(t=bad_t).status_code == 422  # 未严格递增
    assert post(t=[0.5 * i for i in range(n)]).status_code == 422  # t 非整数
    assert post(residuals=[0.0] * (n - 1)).status_code == 422  # 长度不一致
    assert post_raw({**base, "residuals": [float("nan")] + [0.0] * (n - 1)}).status_code == 422
    assert post_raw({**base, "residuals": [float("inf")] + [0.0] * (n - 1)}).status_code == 422
    assert post(min_segment=4).status_code == 422
    assert post(min_segment=101).status_code == 422
    assert post(min_segment=n + 1).status_code == 422  # 超过点数
    assert post(min_segment=5.0).status_code == 422  # 浮点非整数
    assert post(min_segment=True).status_code == 422  # bool 不接受
    assert post(max_changes=-1).status_code == 422
    assert post(max_changes=5).status_code == 422
    assert post(penalty=-0.1).status_code == 422
    assert post_raw({**base, "penalty": float("inf")}).status_code == 422
    assert post_raw({**base, "penalty": float("nan")}).status_code == 422
    assert post(unknown_field=1).status_code == 200  # extra 字段忽略（与 fit 一致）
