from __future__ import annotations

import numpy as np


class NewtonSolveError(RuntimeError):
    pass


def finite_difference_jacobian(func, x: np.ndarray, step: float) -> np.ndarray:
    f0 = np.asarray(func(x), dtype=float)
    jac = np.zeros((f0.size, x.size), dtype=float)
    for i in range(x.size):
        dx = np.zeros_like(x)
        scale = max(1.0, abs(x[i]))
        dx[i] = step * scale
        fp = np.asarray(func(x + dx), dtype=float)
        fm = np.asarray(func(x - dx), dtype=float)
        jac[:, i] = (fp - fm) / (2.0 * dx[i])
    return jac


def newton_raphson_fd(func, x0: np.ndarray, tol: float, max_iter: int, step: float) -> tuple[np.ndarray, int]:
    x = np.asarray(x0, dtype=float).copy()
    for iteration in range(1, max_iter + 1):
        residual = np.asarray(func(x), dtype=float)
        if np.linalg.norm(residual, ord=np.inf) < tol:
            return x, iteration
        jac = finite_difference_jacobian(func, x, step)
        try:
            delta = np.linalg.solve(jac, -residual)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(jac, -residual, rcond=None)
        damping = 1.0
        base_norm = np.linalg.norm(residual, ord=np.inf)
        while damping > 1.0e-3:
            trial = x + damping * delta
            trial_norm = np.linalg.norm(np.asarray(func(trial), dtype=float), ord=np.inf)
            if trial_norm < base_norm:
                x = trial
                break
            damping *= 0.5
        else:
            x = x + delta
    raise NewtonSolveError("Newton-Raphson did not converge within the iteration limit.")
