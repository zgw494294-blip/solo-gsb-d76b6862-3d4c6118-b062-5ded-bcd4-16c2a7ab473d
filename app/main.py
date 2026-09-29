"""FastAPI 应用入口：POST /v1/tide/fit 潮汐调和拟合。"""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from . import tide
from .schemas import (
    DatumShiftRequest,
    DatumShiftResponse,
    TideComponent,
    TideFitRequest,
    TideFitResponse,
    TidePrediction,
)
from .segments import locate_shifts

app = FastAPI(
    title="本地潮汐调和拟合 API",
    version="1.0.0",
    description=(
        "对 20~500 条潮位观测做带 Huber 损失的调和拟合 "
        "c + Σ[a·cos(2πt/P) + b·sin(2πt/P)]，并给出残差与预测。"
    ),
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 错误体只回显字段位置与原因，不回显原始输入。

    （原始输入可能包含 NaN/Infinity 等无法被严格 JSON 序列化的值。）
    """
    detail = [
        {"loc": list(err.get("loc", [])), "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/healthz", tags=["health"])
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/tide/fit",
    response_model=TideFitResponse,
    tags=["tide"],
    summary="Huber 调和拟合并预测潮位",
)
def fit_tide(request: TideFitRequest) -> TideFitResponse:
    # 以首条观测为时间原点；先转 float64 再做减法，避免大整数溢出。
    t = np.asarray(request.t, dtype=np.float64)
    t_rel = t - t[0]
    h = np.asarray(request.h, dtype=np.float64)
    periods = np.asarray(request.periods, dtype=np.float64)

    x = tide.build_design_matrix(t_rel, periods)

    # 设计矩阵必须满列秩；秩亏（如周期在给定采样下混叠）返回 422。
    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"设计矩阵不满列秩：秩为 {rank}，列数为 {x.shape[1]}；"
                "周期可能在当前采样时刻下混叠（线性相关），请调整周期或观测时刻。"
            ),
        )

    try:
        theta, _diagnostics = tide.fit_huber(x, h, request.delta)
    except Exception as exc:  # 数值求解失败属于服务器端问题
        raise HTTPException(status_code=500, detail=f"Huber 拟合失败：{exc}") from exc

    fitted = x @ theta
    residuals = h - fitted

    components = [
        TideComponent(period=float(periods[k]), a=float(theta[1 + 2 * k]), b=float(theta[2 + 2 * k]))
        for k in range(len(request.periods))
    ]

    predict_times = request.predict_times
    if predict_times:
        tp = np.asarray(predict_times, dtype=np.float64) - t[0]
        xp = tide.build_design_matrix(tp, periods)
        predicted = xp @ theta
        predictions = [
            TidePrediction(time=int(predict_times[i]), value=float(predicted[i]))
            for i in range(len(predict_times))
        ]
    else:
        predictions = []

    return TideFitResponse(
        baseline=float(theta[0]),
        components=components,
        residuals=[float(r) for r in residuals],
        predictions=predictions,
    )


@app.post(
    "/v1/tide/datum-shift",
    response_model=DatumShiftResponse,
    tags=["tide"],
    summary="测站基准突变定位（残差分段中位数 L1 分割）",
)
def locate_datum_shift(request: DatumShiftRequest) -> DatumShiftResponse:
    """以 /v1/tide/fit 的逐点 residuals 与同一组 t 定位持续偏移的起点。

    每段偏移取段内残差的较小中位数，最小化「各点到所属偏移的绝对差
    之和 + penalty × 变点数」，并满足最短段长约束；并列时变点少者优先，
    再取变点索引序列字典序最小者。
    """
    result = locate_shifts(
        residuals=request.residuals,
        times=request.t,
        min_segment=request.min_segment,
        max_changes=request.max_changes,
        penalty=request.penalty,
    )
    return DatumShiftResponse(**result)
