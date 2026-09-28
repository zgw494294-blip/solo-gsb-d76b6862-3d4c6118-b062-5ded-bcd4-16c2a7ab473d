"""POST /v1/tide/fit 的端到端测试。"""
from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def post_raw(payload: dict) -> "TestClient":  # type: ignore[name-defined]
    """以严格 / 非严格 JSON 发送（允许 NaN、Infinity token 到达服务端）。"""
    return client.post(
        "/v1/tide/fit",
        content=json.dumps(payload, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )



def make_signal(n: int = 60, step: int = 30, seed: int = 0, outlier_frac: float = 0.0):
    """合成潮汐信号：c=2.5, P=745 分量 (a,b)=(1.2,-0.7), P=1490 分量 (0.5,0.3)。"""
    rng = np.random.default_rng(seed)
    t = (np.arange(n) * step).tolist()
    tt = np.arange(n) * step
    h = (
        2.5
        + 1.2 * np.cos(2 * np.pi * tt / 745.0)
        - 0.7 * np.sin(2 * np.pi * tt / 745.0)
        + 0.5 * np.cos(2 * np.pi * tt / 1490.0)
        + 0.3 * np.sin(2 * np.pi * tt / 1490.0)
        + rng.normal(0.0, 0.02, n)
    )
    if outlier_frac > 0:
        idx = rng.choice(n, size=max(1, int(n * outlier_frac)), replace=False)
        h[idx] += rng.choice([-1.0, 1.0], size=idx.size) * 8.0
    return t, h.tolist()


def test_clean_signal_recovers_parameters():
    t, h = make_signal(seed=1)
    resp = client.post(
        "/v1/tide/fit",
        json={"t": t, "h": h, "P": [745, 1490], "delta": 0.2, "predict_times": [0, 1490]},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert abs(data["baseline"] - 2.5) < 0.03
    comps = {c["period"]: c for c in data["components"]}
    assert set(comps) == {745.0, 1490.0}
    assert abs(comps[745.0]["a"] - 1.2) < 0.03
    assert abs(comps[745.0]["b"] - (-0.7)) < 0.03
    assert abs(comps[1490.0]["a"] - 0.5) < 0.05
    assert abs(comps[1490.0]["b"] - 0.3) < 0.05

    assert len(data["residuals"]) == len(t)
    assert max(abs(r) for r in data["residuals"]) < 0.1

    pred = {p["time"]: p["value"] for p in data["predictions"]}
    assert pred[0] == 0 or pred[0]  # 两个时刻均有结果
    # t=0 处：2.5 + 1.2 + 0.5 = 4.2
    assert abs(pred[0] - 4.2) < 0.06
    # t=1490 处：cos(4π)=1,sin(4π)=0；cos(2π)=1,sin(2π)=0 → 同样 4.2
    assert abs(pred[1490] - 4.2) < 0.06


def test_huber_resists_outliers_unlike_ls():
    """有异常点时，Huber 拟合应明显优于普通最小二乘（不剔除点）。"""
    t, h_bad = make_signal(seed=2, outlier_frac=0.15)
    payload = {"t": t, "h": h_bad, "P": [745], "delta": 0.5, "predict_times": [0]}

    resp = client.post("/v1/tide/fit", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert abs(data["baseline"] - 2.5) < 0.3

    # 对照：直接最小二乘会被异常点带偏
    tt = np.asarray(t, dtype=float)
    x = np.column_stack(
        [np.ones_like(tt), np.cos(2 * np.pi * tt / 745.0), np.sin(2 * np.pi * tt / 745.0)]
    )
    theta_ls, *_ = np.linalg.lstsq(x, np.asarray(h_bad), rcond=None)
    assert abs(theta_ls[0] - 2.5) > abs(data["baseline"] - 2.5)


def test_time_origin_is_first_observation():
    """首条观测 t0 非零时，内部以 t-t0 计算；结果与从 0 开始一致。"""
    t0, h0 = make_signal(n=40, seed=3)
    offset = 10_000
    resp = client.post(
        "/v1/tide/fit",
        json={"t": [x + offset for x in t0], "h": h0, "P": [745], "delta": 0.2},
    )
    resp_base = client.post(
        "/v1/tide/fit", json={"t": t0, "h": h0, "P": [745], "delta": 0.2}
    )
    assert resp.status_code == resp_base.status_code == 200
    a = resp.json()
    b = resp_base.json()
    assert abs(a["baseline"] - b["baseline"]) < 1e-10
    assert abs(a["components"][0]["a"] - b["components"][0]["a"]) < 1e-10
    assert abs(a["components"][0]["b"] - b["components"][0]["b"]) < 1e-10


def test_rank_deficient_returns_422():
    """周期在整数分钟采样下完全混叠（Nyquist）→ 设计矩阵秩亏 → 422。"""
    n = 30
    t = (np.arange(n) * 30).tolist()
    h = np.ones(n).tolist()
    # P=60 与 P=-60 的余弦在 30 分钟采样下相同；这里用两个混叠周期：
    # cos(2π t/60) 在 t=30k 时为 cos(kπ)=(-1)^k；
    # 周期 P=20 时 cos(2π·30k/20)=cos(3kπ)=(-1)^k → 完全相同。
    resp = client.post(
        "/v1/tide/fit", json={"t": t, "h": h, "P": [60, 20], "delta": 1.0}
    )
    assert resp.status_code == 422
    assert "满列秩" in resp.json()["detail"]


def test_validation_errors_are_422():
    base_t, base_h = make_signal(n=25, seed=4)
    base = {"t": base_t, "h": base_h, "P": [745], "delta": 1.0}

    def post(**overrides):
        payload = {**base, **overrides}
        return client.post("/v1/tide/fit", json=payload)

    assert post(t=base_t[:19], h=base_h[:19]).status_code == 422  # 少于 20
    bad_t = list(base_t)
    bad_t[5] = bad_t[4]
    assert post(t=bad_t).status_code == 422  # 非严格递增
    assert post_raw({**base, "h": [float("nan")] + base_h[1:]}).status_code == 422
    assert post_raw({**base, "h": [float("inf")] + base_h[1:]}).status_code == 422
    assert post(P=[745, 745]).status_code == 422  # 周期重复
    assert post(P=[0]).status_code == 422  # 周期非正
    assert post(P=[-10]).status_code == 422
    assert post(delta=0).status_code == 422  # delta 非正
    assert post(delta=-1.0).status_code == 422
    assert post(h=base_h[:-1]).status_code == 422  # t/h 长度不一致
    assert post(P=[745, 746, 747, 748, 749]).status_code == 422  # 超过 4 个周期
    assert post(t=base_t, predict_times=[1.5]).status_code == 422  # 预测时刻非整数
    long_t = list(range(0, 501))
    assert post(t=long_t, h=[1.0] * 501).status_code == 422  # 超过 500 条
