"""POST /v1/tide/change-points 的端到端测试，以及与 /v1/tide/fit 的串接。"""
from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.segments import precompute_segment_costs, solve_segmentation

client = TestClient(app)

FIT_PATH = "/v1/tide/fit"
CP_PATH = "/v1/tide/change-points"


def post_raw(path: str, payload: dict) -> "TestClient":  # type: ignore[name-defined]
    """以非严格 JSON 发送（允许 NaN、Infinity token 到达服务端）。"""
    return client.post(
        path,
        content=json.dumps(payload, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )


def make_shifted_signal(n: int = 80, step: int = 30, jump_at: int = 40, shift: float = 1.5):
    """合成潮汐信号：在 jump_at 处发生持续性基准突变，另有单尖峰异常。"""
    rng = np.random.default_rng(7)
    t = (np.arange(n) * step).tolist()
    tt = np.arange(n) * step
    h = (
        2.5
        + 1.2 * np.cos(2 * np.pi * tt / 745.0)
        - 0.7 * np.sin(2 * np.pi * tt / 745.0)
        + rng.normal(0.0, 0.02, n)
    )
    h[jump_at:] += shift
    h[12] += 6.0  # 单尖峰：不应被定位为持续突变
    return t, h.tolist()


def test_end_to_end_fit_then_change_points():
    """先调 /v1/tide/fit 取 residuals，再串接突变定位，应只检出真正的突变点。

    未建模的持续阶跃会整体带偏调和拟合（两段残差存在公共偏置），因此
    断言对公共偏置稳健：突变位置正确、两段偏移之差等于阶跃量、校正后
    各段内的较小中位数归零、尖峰点不产生额外变点。
    """
    t, h = make_shifted_signal()
    fit = client.post(FIT_PATH, json={"t": t, "h": h, "P": [745], "delta": 1.0})
    assert fit.status_code == 200, fit.text
    residuals = fit.json()["residuals"]

    resp = client.post(
        CP_PATH,
        json={
            "t": t,
            "residuals": residuals,
            "min_segment": 10,
            "max_changes": 4,
            "penalty": 5.0,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["change_indices"] == [40]
    assert data["change_times"] == [t[40]]

    segments = data["segments"]
    assert [(s["start"], s["end"]) for s in segments] == [(0, 40), (40, 80)]
    assert segments[0]["change_time"] is None
    assert segments[1]["change_time"] == t[40]

    # 偏移之差 = 持续阶跃量（公共拟合偏置相消）。
    assert abs((data["offsets"][1] - data["offsets"][0]) - 1.5) < 0.2

    assert len(data["corrected_residuals"]) == len(t)
    corrected = np.asarray(data["corrected_residuals"])
    # 各段以自身较小中位数为偏移，校正后段内较小中位数严格为 0。
    assert np.sort(corrected[:40])[19] == 0.0
    assert np.sort(corrected[40:])[19] == 0.0

    sad = float(np.sum(np.abs(corrected)))
    assert abs(data["total_cost"] - sad - 5.0) < 1e-9


def test_no_change_with_zero_max_changes_or_huge_penalty():
    t, h = make_shifted_signal()
    residuals = client.post(
        FIT_PATH, json={"t": t, "h": h, "P": [745], "delta": 0.5}
    ).json()["residuals"]

    base = {"t": t, "residuals": residuals, "min_segment": 10}

    resp0 = client.post(CP_PATH, json={**base, "max_changes": 0, "penalty": 0.0})
    assert resp0.status_code == 200
    data0 = resp0.json()
    assert data0["change_indices"] == []
    assert data0["change_times"] == []
    assert [(s["start"], s["end"]) for s in data0["segments"]] == [(0, 80)]
    assert data0["segments"][0]["change_time"] is None
    assert len(data0["offsets"]) == 1

    resp_pen = client.post(CP_PATH, json={**base, "max_changes": 4, "penalty": 1e9})
    data_pen = resp_pen.json()
    assert data_pen["change_indices"] == []
    # 无变点时总目标值就是整体绝对差之和
    assert abs(data_pen["total_cost"] - float(np.sum(np.abs(np.asarray(residuals) - data_pen["offsets"][0])))) < 1e-9


def test_multiple_changes_and_segment_coverage():
    """三个常值块：各段连续覆盖全序列，[start,end) 首尾相接。"""
    n = 60
    t = (np.arange(n) * 15).tolist()
    residuals = np.concatenate(
        [np.full(20, 0.1), np.full(20, 2.0), np.full(20, -1.0)]
    ).tolist()
    resp = client.post(
        CP_PATH,
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
    assert data["change_indices"] == [20, 40]
    assert data["change_times"] == [t[20], t[40]]
    assert [(s["start"], s["end"]) for s in data["segments"]] == [
        (0, 20),
        (20, 40),
        (40, 60),
    ]
    assert [s["change_time"] for s in data["segments"]] == [None, t[20], t[40]]
    assert data["offsets"] == [0.1, 2.0, -1.0]
    assert max(abs(x) for x in data["corrected_residuals"]) < 1e-12
    assert abs(data["total_cost"]) < 1e-12


def test_lower_median_for_even_segments():
    """偶数长度段偏移取较小中位数（排序后下标 (m-1)//2）。"""
    n = 20
    t = list(range(n))
    # 两段各 10 个连续整数（偶数长度），段间跳跃足够大。
    residuals = [float(v) for v in range(0, 10)] + [float(v) for v in range(100, 110)]
    # penalty 抑制块内切分（单调 10 点切半仅省 13）但不抑制块间跳变（省 950）。
    resp = client.post(
        CP_PATH,
        json={
            "t": t,
            "residuals": residuals,
            "min_segment": 5,
            "max_changes": 4,
            "penalty": 20.0,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["change_indices"] == [10]
    # 0..9 的两个中位数为 4、5 → 较小中位数 4；100..109 → 104。
    assert data["segments"][0]["offset"] == 4.0
    assert data["segments"][1]["offset"] == 104.0
    # 连续 10 个整数到较小中位数的 SAD：
    # 上半和(5+6+7+8+9) - 下半和(0+1+2+3+4) = 35 - 10 = 25，两段共 50，
    # 总目标 = 50 + 1 个变点 × 20 = 70。
    assert data["total_cost"] == 70.0


def test_fewer_changes_preferred_on_tie():
    """penalty=0 且多切一段仍为 0 代价时，并列先选变点少者。"""
    n = 20
    t = (np.arange(n) * 30).tolist()
    residuals = [1.0] * 10 + [4.0] * 10
    resp = client.post(
        CP_PATH,
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
    # 任意把常值块切成 5+5 都是 0 代价，但先选变点最少：仅 1 个变点。
    assert data["change_indices"] == [10]


def test_segment_length_constraint_respected():
    """min_segment 限制：L=10、n=20 时唯一可行的单变点只能在索引 10。"""
    n = 20
    t = list(range(n))
    # 真实跳变在 8，但 L=10 不允许 [0,8)；最优可行边界是 10。
    residuals = [0.0] * 8 + [3.0] * 12
    resp = client.post(
        CP_PATH,
        json={
            "t": t,
            "residuals": residuals,
            "min_segment": 10,
            "max_changes": 4,
            "penalty": 0.0,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["change_indices"] == [10]


def test_min_segment_equals_point_count_is_one_segment():
    n = 20
    t = list(range(n))
    residuals = [float(i % 3) for i in range(n)]
    resp = client.post(
        CP_PATH,
        json={
            "t": t,
            "residuals": residuals,
            "min_segment": n,  # 允许“等于点数”，此时只能 1 段
            "max_changes": 4,
            "penalty": 0.0,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["change_indices"] == []
    assert [(s["start"], s["end"]) for s in data["segments"]] == [(0, n)]


def test_lex_tie_break_via_dp_unit():
    """直接对 DP 注入代价表：同变点数同代价时选索引元组字典序最小者。"""
    n, L = 15, 5
    cost = precompute_segment_costs(np.zeros(n), L)
    for i in range(n):
        for j in range(n + 1):
            if j - i >= L:
                cost[i][j] = 100.0
    cost[0][n] = 10.0
    cost[0][5] = 3.0
    cost[5][n] = 3.0
    cost[0][10] = 3.0
    cost[10][n] = 3.0
    cost[5][10] = 3.0

    boundaries, objective = solve_segmentation(cost, n, L, max_segments=3, penalty=0.0)
    assert boundaries == (5,)
    assert objective == 6.0

    # penalty=5：1 段（10）优于 2 段（6+5=11）
    boundaries_p, objective_p = solve_segmentation(
        cost, n, L, max_segments=3, penalty=5.0
    )
    assert boundaries_p == ()
    assert objective_p == 10.0


def test_validation_errors_are_422():
    n = 25
    t = (np.arange(n) * 30).tolist()
    residuals = np.zeros(n).tolist()
    base = {
        "t": t,
        "residuals": residuals,
        "min_segment": 5,
        "max_changes": 2,
        "penalty": 1.0,
    }

    def post(**overrides):
        payload = {**base, **overrides}
        return client.post(CP_PATH, json=payload)

    assert post(t=t[:19], residuals=residuals[:19]).status_code == 422  # 少于 20
    long_t = list(range(501))
    assert post(t=long_t, residuals=[0.0] * 501).status_code == 422  # 超过 500
    bad_t = list(t)
    bad_t[5] = bad_t[4]
    assert post(t=bad_t).status_code == 422  # 未严格递增
    assert post(t=[1.5] * n).status_code == 422  # t 非整数
    assert post(t=[True] * n).status_code == 422  # bool 不算整数
    assert post(residuals=residuals[:-1]).status_code == 422  # 长度不一致
    assert post_raw(CP_PATH, {**base, "residuals": [float("nan")] + residuals[1:]}).status_code == 422
    assert post_raw(CP_PATH, {**base, "residuals": [float("inf")] + residuals[1:]}).status_code == 422
    assert post(min_segment=4).status_code == 422  # < 5
    assert post(min_segment=101).status_code == 422  # > 100
    assert post(min_segment=n + 1).status_code == 422  # 超过点数
    assert post(min_segment=5.0).status_code == 422  # 非整数
    assert post(max_changes=-1).status_code == 422
    assert post(max_changes=5).status_code == 422
    assert post(penalty=-0.1).status_code == 422
    assert post_raw(CP_PATH, {**base, "penalty": float("inf")}).status_code == 422
    assert post_raw(CP_PATH, {**base, "penalty": float("nan")}).status_code == 422

    # 缺字段
    for key in ("t", "residuals", "min_segment", "max_changes", "penalty"):
        payload = {k: v for k, v in base.items() if k != key}
        assert client.post(CP_PATH, json=payload).status_code == 422, key
