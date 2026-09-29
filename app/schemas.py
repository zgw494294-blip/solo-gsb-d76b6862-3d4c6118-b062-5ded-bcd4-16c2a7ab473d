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


class ChangePointRequest(BaseModel):
    """POST /v1/tide/change-points 请求体。

    输入直接来自 /v1/tide/fit 的输出：同一批观测的整数分钟时刻 t 与
    逐点 residuals。残差中若存在若干处持续性的整体平移（测站基准突变），
    用带最小段长与变点惩罚的 L1 分段把突变位置找出来。
    """

    t: list[int] = Field(..., description="观测时刻（整数分钟），严格递增，20~500 条")
    residuals: list[float] = Field(
        ..., description="逐点残差（来自 /v1/tide/fit），与 t 等长，全部为有限数"
    )
    min_segment: int = Field(
        ..., description="最短段长（点数），5~100，且不超过总点数"
    )
    max_changes: int = Field(..., description="最多允许的变点数，0~4")
    penalty: float = Field(
        ..., description="每个变点的目标函数惩罚，非负有限数"
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

    @field_validator("residuals")
    @classmethod
    def _validate_residuals(cls, v: list[float]) -> list[float]:
        if any(not math.isfinite(float(x)) for x in v):
            raise ValueError("residuals 必须全部为有限数（不允许 NaN / Inf）")
        return [float(x) for x in v]

    @field_validator("min_segment", mode="before")
    @classmethod
    def _validate_min_segment(cls, v: Any) -> Any:
        # 先于强转拦截 5.0 这类 float 与 bool（pydantic 默认会把整数
        # 值的 float 强转为 int）。
        if not cls._is_strict_int(v):
            raise ValueError("min_segment 必须为整数")
        if not 5 <= v <= 100:
            raise ValueError("min_segment 必须在 5~100 之间")
        return v

    @field_validator("max_changes", mode="before")
    @classmethod
    def _validate_max_changes(cls, v: Any) -> Any:
        if not cls._is_strict_int(v):
            raise ValueError("max_changes 必须为整数")
        if not 0 <= v <= 4:
            raise ValueError("max_changes 必须在 0~4 之间")
        return v
    @field_validator("penalty")
    @classmethod
    def _validate_penalty(cls, v: float) -> float:
        if not math.isfinite(float(v)) or v < 0.0:
            raise ValueError("penalty 必须为非负有限数")
        return float(v)

    @model_validator(mode="after")
    def _validate_lengths(self) -> "ChangePointRequest":
        if len(self.t) != len(self.residuals):
            raise ValueError("t 与 residuals 长度必须相同")
        if self.min_segment > len(self.t):
            raise ValueError("min_segment 不能超过总点数")
        return self


class ChangePointSegment(BaseModel):
    """单个残差分段。"""

    start: int = Field(..., description="段首点下标（含）")
    end: int = Field(..., description="段尾点下标（不含）")
    change_time: int | None = Field(
        ..., description="变点（新段首点）的整数分钟时刻；首段为 null"
    )
    offset: float = Field(..., description="该段基准偏移（段内残差的较小中位数）")


class ChangePointResponse(BaseModel):
    """基准突变定位结果。"""

    segments: list[ChangePointSegment] = Field(
        ..., description="各段 [start, end) 索引、变点时刻与偏移"
    )
    change_indices: list[int] = Field(
        ..., description="变点所在新段首点的索引（即第 2 段起的 start）"
    )
    change_times: list[int] = Field(
        ..., description="各变点（新段首点）的整数分钟时刻"
    )
    offsets: list[float] = Field(..., description="各段偏移，与 segments 同序")
    corrected_residuals: list[float] = Field(
        ..., description="校正后逐点残差（原残差减去所属段偏移），与 t 等长"
    )
    total_cost: float = Field(
        ..., description="总目标值：各点到所属偏移绝对差之和 + penalty × 变点数"
    )
