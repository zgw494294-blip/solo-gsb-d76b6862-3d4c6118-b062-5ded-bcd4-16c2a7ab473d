"""请求 / 响应数据模型与输入校验。"""
from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class TideFitRequest(BaseModel):
    """POST /v1/tide/fit 请求体。

    时间 t 以整数分钟给出，必须严格递增；以首条观测作为时间原点，
    因此内部拟合使用的时刻为 t - t[0]。
    """

    t: list[int] = Field(..., description="观测时刻（整数分钟），严格递增，20~500 条")
    h: list[float] = Field(..., description="潮位观测，长度与 t 相同，全部为有限数")
    periods: list[float] = Field(
        ..., alias="P", description="调和分量周期（分钟），1~4 个，互异正数"
    )
    delta: float = Field(..., description="Huber 损失阈值，必须为正数")
    predict_times: list[int] = Field(
        default_factory=list,
        description="需要预测潮位的整数分钟时刻（与 t 同一时间坐标）",
    )

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @staticmethod
    def _is_strict_int(x: Any) -> bool:
        return isinstance(x, int) and not isinstance(x, bool)

    @field_validator("t")
    @classmethod
    def _validate_t(cls, v: list[int]) -> list[int]:
        if not 20 <= len(v) <= 500:
            raise ValueError("t 的长度必须在 20~500 之间")
        if any(not cls._is_strict_int(x) for x in v):
            raise ValueError("t 必须全部为整数")
        if any(b <= a for a, b in zip(v, v[1:])):
            raise ValueError("t 必须严格递增")
        return v

    @field_validator("h")
    @classmethod
    def _validate_h(cls, v: list[float]) -> list[float]:
        if any(not math.isfinite(float(x)) for x in v):
            raise ValueError("h 必须全部为有限数（不允许 NaN / Inf）")
        return v

    @field_validator("periods")
    @classmethod
    def _validate_periods(cls, v: list[float]) -> list[float]:
        if not 1 <= len(v) <= 4:
            raise ValueError("P 必须包含 1~4 个周期")
        if any(not math.isfinite(float(p)) or p <= 0.0 for p in v):
            raise ValueError("P 中的周期必须全部为有限正数")
        if len({float(p) for p in v}) != len(v):
            raise ValueError("P 中的周期必须互异")
        return [float(p) for p in v]

    @field_validator("delta")
    @classmethod
    def _validate_delta(cls, v: float) -> float:
        if not math.isfinite(float(v)) or v <= 0.0:
            raise ValueError("delta 必须为有限正数")
        return float(v)

    @field_validator("predict_times")
    @classmethod
    def _validate_predict_times(cls, v: list[int]) -> list[int]:
        if any(not cls._is_strict_int(x) for x in v):
            raise ValueError("predict_times 必须全部为整数")
        return v

    @model_validator(mode="after")
    def _validate_lengths(self) -> "TideFitRequest":
        if len(self.t) != len(self.h):
            raise ValueError("t 与 h 长度必须相同")
        return self


class TideComponent(BaseModel):
    """单个调和分量的拟合系数。"""

    period: float = Field(..., description="周期 P（分钟）")
    a: float = Field(..., description="cos(2πt/P) 系数")
    b: float = Field(..., description="sin(2πt/P) 系数")


class TidePrediction(BaseModel):
    """单个预测时刻的潮位。"""

    time: int = Field(..., description="预测时刻（整数分钟，与请求同坐标）")
    value: float = Field(..., description="预测潮位")


class TideFitResponse(BaseModel):
    """拟合结果。"""

    baseline: float = Field(..., description="基准潮位 c")
    components: list[TideComponent] = Field(..., description="各周期的调和系数")
    residuals: list[float] = Field(..., description="逐点残差（观测 - 拟合），与 t 等长")
    predictions: list[TidePrediction] = Field(
        ..., description="predict_times 对应的预测潮位"
    )
