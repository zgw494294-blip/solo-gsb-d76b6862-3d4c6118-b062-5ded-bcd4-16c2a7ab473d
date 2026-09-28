"""潮汐调和模型与 Huber 损失拟合。

模型（时间原点取首条观测）::

    h(t) = c + Σ_k [ a_k cos(2π (t - t0) / P_k)
                      + b_k sin(2π (t - t0) / P_k) ]

参数顺序：theta = [c, a_1, b_1, a_2, b_2, ...]。

损失使用全部观测、不剔除任何点::

    L(theta) = Σ_i H_delta(h_i - X_i θ),
    H_delta(r) = 0.5 r²            (|r| ≤ delta)
                 delta |r| - 0.5 delta²  (|r| > delta)

求解策略：Huber 损失为凸函数。先用迭代加权最小二乘（IRLS，即其
MM 二次上界迭代）快速逼近，再用 L-BFGS-B（解析梯度）做最终收敛，
保证得到全局最小值。
"""
from __future__ import annotations

import numpy as np
from scipy import optimize

TAU2PI = 2.0 * np.pi


def build_design_matrix(t_rel: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """构建设计矩阵 X，列依次为 [1, cos, sin, cos, sin, ...]。

    t_rel: 已相对首条观测平移的时刻（分钟），float64。
    """
    columns = [np.ones_like(t_rel, dtype=np.float64)]
    for period in periods:
        angle = TAU2PI * t_rel / float(period)
        columns.append(np.cos(angle))
        columns.append(np.sin(angle))
    return np.column_stack(columns)


def huber_loss(theta: np.ndarray, x: np.ndarray, y: np.ndarray, delta: float) -> float:
    residual = y - x @ theta
    abs_r = np.abs(residual)
    quadratic = 0.5 * residual**2
    linear = delta * abs_r - 0.5 * delta**2
    return float(np.sum(np.where(abs_r <= delta, quadratic, linear)))


def huber_gradient(theta: np.ndarray, x: np.ndarray, y: np.ndarray, delta: float) -> np.ndarray:
    """∂L/∂θ = Xᵀ ρ，其中 ρ_i = -clip(r_i, -δ, δ)。"""
    residual = y - x @ theta
    clipped = np.clip(residual, -delta, delta)
    return x.T @ (-clipped)


def _irls(
    x: np.ndarray,
    y: np.ndarray,
    delta: float,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[np.ndarray, bool]:
    """迭代加权最小二乘（Huber 的 MM 算法）。

    每步最小化 Σ w_i (X_i θ - y_i)²，其中
    w_i = 1（|r_i| ≤ δ），w_i = δ / |r_i|（|r_i| > δ）。
    损失单调不增，收敛到全局最优。
    """
    theta, *_ = np.linalg.lstsq(x, y, rcond=None)
    converged = False
    for _ in range(max_iter):
        residual = y - x @ theta
        abs_r = np.abs(residual)
        weights = np.where(abs_r > delta, delta / np.maximum(abs_r, 1e-300), 1.0)
        sw = np.sqrt(weights)
        theta_new, *_ = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)
        step = float(np.max(np.abs(theta_new - theta)))
        theta = theta_new
        if step <= tol * (1.0 + float(np.max(np.abs(theta)))):
            converged = True
            break
    return theta, converged


def fit_huber(x: np.ndarray, y: np.ndarray, delta: float) -> tuple[np.ndarray, dict]:
    """最小化全部观测上的 Huber 损失，返回 (参数, 诊断信息)。"""
    theta0, irls_converged = _irls(x, y, delta)

    result = optimize.minimize(
        huber_loss,
        theta0,
        args=(x, y, delta),
        jac=huber_gradient,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-9},
    )

    if not result.success:
        # 极小概率的非收敛：换一种拟牛顿法再试，仍失败则交由上层报 500。
        result = optimize.minimize(
            huber_loss,
            theta0,
            args=(x, y, delta),
            jac=huber_gradient,
            method="CG",
            options={"maxiter": 10000, "gtol": 1e-9},
        )

    diagnostics = {
        "loss": float(result.fun),
        "irls_converged": irls_converged,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }
    return np.asarray(result.x, dtype=np.float64), diagnostics
