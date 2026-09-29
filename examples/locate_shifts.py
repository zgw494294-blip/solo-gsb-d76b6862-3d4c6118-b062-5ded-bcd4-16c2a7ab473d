"""可执行示例：先调用 /v1/tide/fit 做调和拟合，再把逐点 residuals 交给
/v1/tide/change-points 定位测站基准突变，全程仅依赖标准库。

运行（需先启动服务，默认 8000 端口）：

    docker compose up --build        # 或 uvicorn app.main:app --port 8000
    python examples/locate_shifts.py
    BASE_URL=http://127.0.0.1:8000 python examples/locate_shifts.py
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    # 合成观测：潮汐信号 + 观测噪声，在第 40 点（1200 分钟）处基准整体抬升 1.5，
    # 另含一个传感器尖峰（持续突变定位应忽略这种单点异常）。
    n, step, jump_at, jump_time, shift = 80, 30, 40, 1200, 1.5
    tt = np.arange(n) * step
    rng = np.random.default_rng(7)
    h = (
        2.5
        + 1.2 * np.cos(2 * np.pi * tt / 745.0)
        - 0.7 * np.sin(2 * np.pi * tt / 745.0)
        + rng.normal(0.0, 0.02, n)
    )
    h[jump_at:] += shift
    h[12] += 6.0
    t = tt.tolist()

    # 第一次调用：Huber 调和拟合，取逐点 residuals。
    fit = post(
        "/v1/tide/fit",
        {"t": t, "h": h.tolist(), "P": [745], "delta": 1.0},
    )

    # 第二次调用：以原接口返回的 residuals 与严格递增的整数分钟 t 为输入，
    # 定位持续偏移从何时开始。
    result = post(
        "/v1/tide/change-points",
        {
            "t": t,
            "residuals": fit["residuals"],
            "min_segment": 10,   # 每段至少 10 点（5~100 且不超过点数）
            "max_changes": 4,    # 最多 4 个变点
            "penalty": 5.0,      # 每个变点的目标函数惩罚
        },
    )

    print(f"检测到 {len(result['change_indices'])} 个基准突变：")
    for seg in result["segments"]:
        head = (
            "首段"
            if seg["change_time"] is None
            else f"突变始于 t={seg['change_time']} 分钟（索引 {seg['start']}）"
        )
        print(
            f"  [{seg['start']:>2}, {seg['end']:>2})  偏移={seg['offset']:+.4f}  {head}"
        )
    print(f"总目标值（绝对差之和 + penalty×变点数）：{result['total_cost']:.4f}")

    assert result["change_times"] == [jump_time], result["change_times"]
    jump_size = result["offsets"][1] - result["offsets"][0]
    print(f"恢复的基准跳变量：{jump_size:+.4f}（真值 {shift:+.1f}）")
    assert abs(jump_size - shift) < 0.2


if __name__ == "__main__":
    main()
