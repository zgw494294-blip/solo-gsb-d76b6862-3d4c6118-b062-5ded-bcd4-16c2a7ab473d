"""串接两次 API 调用的可执行示例：先调和拟合，再定位测站基准突变。

用法（需先在本机启动服务，端口 8000，见 README）::

    python examples/chained_datum_shift.py
    python examples/chained_datum_shift.py --base-url http://localhost:8000

脚本仅依赖 Python 标准库（urllib）；合成数据时借助运行服务同一环境中
的 numpy（服务本身已依赖）。
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request

import numpy as np


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # 422 等错误：回显服务端 detail
        detail = exc.read().decode("utf-8")
        raise SystemExit(f"HTTP {exc.code} from {url}:\n{detail}") from exc


def make_observations(n: int = 60, step: int = 30, shift_index: int = 35) -> tuple[list, list]:
    """合成 60 点潮位观测，从 shift_index 起基准整体抬升 1.0。"""
    rng = np.random.default_rng(2026)
    tt = np.arange(n) * step
    h = (
        2.5
        + 1.2 * np.cos(2 * np.pi * tt / 745.0)
        - 0.7 * np.sin(2 * np.pi * tt / 745.0)
        + rng.normal(0.0, 0.02, n)
    )
    h[shift_index:] += 1.0  # 测站基准突变（水尺零点重设）
    return tt.tolist(), [round(float(v), 4) for v in h]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--min-segment", type=int, default=10)
    parser.add_argument("--max-changes", type=int, default=2)
    parser.add_argument(
        "--penalty",
        type=float,
        default=1.5,
        help="每变点惩罚；调和模型会部分吸收阶跃、在残差中留下斜坡，"
        "取 1.5 可抑制斜坡诱发的多余碎切",
    )
    args = parser.parse_args()

    t, h = make_observations()

    # 第一次调用：调和拟合，拿到逐点 residuals
    fit = post_json(
        f"{args.base_url}/v1/tide/fit",
        {
            "t": t,
            "h": h,
            "P": [745],
            "delta": 0.2,
            "predict_times": [],
        },
    )
    print("== /v1/tide/fit ==")
    print(f"baseline = {fit['baseline']:.4f}")

    # 第二次调用：把 residuals 与同一组 t 送去定位基准突变
    shifts = post_json(
        f"{args.base_url}/v1/tide/datum-shift",
        {
            "t": t,
            "residuals": fit["residuals"],
            "min_segment": args.min_segment,
            "max_changes": args.max_changes,
            "penalty": args.penalty,
        },
    )
    print("\n== /v1/tide/datum-shift ==")
    for cp in shifts["change_points"]:
        print(
            f"检测到基准突变：新段首点 index={cp['index']}，"
            f"时刻 t={cp['time']} 分钟"
        )
    for seg, offset in zip(shifts["segments"], shifts["offsets"]):
        print(
            f"  段 [{seg['start']:>3}, {seg['end']:>3})  "
            f"t=[{t[seg['start']]}, {t[seg['end'] - 1]}]（末点含）  偏移 = {offset:+.4f}"
        )
    print(f"总目标值 = {shifts['objective']:.4f}")

    residual_scale = math.sqrt(sum(r * r for r in shifts["corrected_residuals"]) / len(t))
    print(f"校正后残差 RMS = {residual_scale:.4f}")
    print("（真实数据中还含调和模型未吸收的潮汐失配；本合成例的观测噪声为 0.02）")


if __name__ == "__main__":
    main()
