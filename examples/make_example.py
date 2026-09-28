"""生成 examples/request.json 中使用的示例观测（含一个异常点，由 Huber 损失稳健处理）。"""
import json
import pathlib

import numpy as np

rng = np.random.default_rng(42)
n = 30
step = 30  # 分钟
t = (np.arange(n) * step).tolist()
tt = np.arange(n) * step
h = (
    2.6
    + 1.3 * np.cos(2 * np.pi * tt / 745.0)
    - 0.6 * np.sin(2 * np.pi * tt / 745.0)
    + 0.25 * np.cos(2 * np.pi * tt / 1490.0)
    + rng.normal(0.0, 0.03, n)
)
h[12] += 6.0  # 传感器尖峰异常；不预先剔除，交给 Huber 损失

payload = {
    "t": t,
    "h": [round(float(v), 2) for v in h],
    "P": [745, 1490],
    "delta": 0.5,
    "predict_times": [900, 1440],
}
path = pathlib.Path(__file__).with_name("request.json")
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("written:", path)
