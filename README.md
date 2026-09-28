# 本地潮汐调和拟合 API

基于 **Python + FastAPI + NumPy + SciPy** 的本地服务，对 20～500 条潮位观测做
**Huber 稳健调和拟合**，并返回基准潮位、各周期余弦/正弦系数、逐点残差与指定时刻的预测潮位。

## 模型

以**首条观测为时间原点**（内部时刻为 `t - t[0]`），拟合

```
h(t) = c + Σ_k [ a_k · cos(2π (t - t0) / P_k)
               + b_k · sin(2π (t - t0) / P_k) ]
```

- `c`：基准潮位（baseline）
- `P_k`：1～4 个互异的正数周期（分钟）
- `a_k, b_k`：对应周期的余弦、正弦系数

使用**全部观测**最小化 Huber 损失，**不预先剔除异常点**：

```
Hδ(r) = 0.5·r²                     |r| ≤ δ
        δ·|r| - 0.5·δ²             |r| >  δ
```

求解：迭代加权最小二乘（IRLS / MM）初始化 + SciPy `L-BFGS-B`（解析梯度）收敛。
Huber 损失为凸函数，可得到全局最优解。

## 一键启动（Docker Compose，端口 8000）

```bash
docker compose up --build
```

服务启动后：

- 接口地址：`http://localhost:8000/v1/tide/fit`
- 交互式文档（Swagger UI）：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/healthz`

停止：`Ctrl+C`（后台运行用 `docker compose up -d --build`，停止用 `docker compose down`）。

也可直接用 Docker：

```bash
docker build -t tide-harmonic-api .
docker run --rm -p 8000:8000 tide-harmonic-api
```

## 本地（无 Docker）运行

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 请求

`POST /v1/tide/fit`，`Content-Type: application/json`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `t` | integer[] | 观测时刻（整数分钟），严格递增，**20～500** 条 |
| `h` | number[] | 潮位观测，与 `t` 等长，全部为**有限数**（不允许 NaN/Inf） |
| `P` | number[] | 周期（分钟），**1～4** 个，互异正数 |
| `delta` | number | Huber 阈值，正数 |
| `predict_times` | integer[] | 可选；需要预测的整数分钟时刻（与 `t` 同一时间坐标，可超出观测范围） |

### curl 示例

```bash
curl -X POST http://localhost:8000/v1/tide/fit \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

或内联：

```bash
curl -X POST http://localhost:8000/v1/tide/fit \
  -H 'Content-Type: application/json' \
  -d '{
    "t": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270,
          300, 330, 360, 390, 420, 450, 480, 510, 540, 570,
          600, 630, 660, 690, 720, 750, 780, 810, 840, 870],
    "h": [4.16, 3.93, 3.71, 3.39, 2.94, 2.58, 2.25, 1.90, 1.62, 1.38,
          1.29, 1.24, 7.26, 1.42, 1.60, 1.83, 2.17, 2.46, 2.85, 3.13,
          3.40, 3.59, 3.78, 3.78, 3.74, 3.61, 3.44, 3.17, 2.86, 2.52],
    "P": [745, 1490],
    "delta": 0.5,
    "predict_times": [900, 1440]
  }'
```

### Python 示例

```python
import requests

r = requests.post("http://localhost:8000/v1/tide/fit", json={
    "t": list(range(0, 900, 30)),
    "h": h_values,
    "P": [745, 1490],
    "delta": 0.5,
    "predict_times": [900, 1440],
})
r.raise_for_status()
print(r.json())
```

## 响应

```json
{
  "baseline": 2.6737,
  "components": [
    {"period": 745.0,  "a": 1.2134, "b": -0.6108},
    {"period": 1490.0, "a": 0.2582, "b": -0.0831}
  ],
  "residuals": [0.0146, -0.0113, "...共 len(t) 个，残差 = 观测 - 拟合",
                5.9267, "...异常点本身的大残差同样保留，不剔除"],
  "predictions": [
    {"time": 900,  "value": 2.2457},
    {"time": 1440, "value": 4.3007}
  ]
}
```

- `baseline`：基准潮位 `c`
- `components`：各周期（与请求中 `P` 同序）的 `a`、`b`
- `residuals`：逐点残差 `h - ĥ`，与 `t` 等长（含异常点本身的残差，不会被剔除）
- `predictions`：`predict_times` 对应预测潮位；未提供时为空数组

## 错误处理（HTTP 422）

以下情况返回 `422 Unprocessable Entity`，错误体 `detail` 中给出字段位置与原因：

- `t` 数量不在 20～500、非整数、未严格递增
- `h` 含 NaN/Infinity 或与 `t` 长度不一致
- `P` 数量不在 1～4、含非正数/非有限数、周期不互异
- `delta` 非正或非有限
- `predict_times` 含非整数
- **设计矩阵不满列秩**（例如某两个周期在当前整数分钟采样下混叠、
  或某周期在采样格点上恒为常数）——请调整周期或观测时刻

## 测试

```bash
pip install pytest httpx
pytest -q
```

测试覆盖：干净信号参数恢复、含异常点时 Huber 相对最小二乘的稳健性、
时间原点平移不变性、秩亏 422 以及各类非法输入 422。

## 项目结构

```
app/
  main.py      # FastAPI 路由、422 校验、秩检查
  schemas.py   # Pydantic 请求/响应模型
  tide.py      # 设计矩阵、Huber 损失、IRLS + L-BFGS-B 求解
examples/      # 请求示例
tests/         # pytest 端到端测试
Dockerfile
docker-compose.yml
requirements.txt
```
