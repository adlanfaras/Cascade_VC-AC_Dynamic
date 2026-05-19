from __future__ import annotations

import numpy as np


class NewtonSolveError(RuntimeError):
    pass


def finite_difference_jacobian(
    func,
    x: np.ndarray,
    step: float,
    f0: np.ndarray | None = None,
    scheme: str = "central",
) -> np.ndarray:
    if f0 is None:
        f0 = np.asarray(func(x), dtype=float)
    jac = np.zeros((f0.size, x.size), dtype=float)
    use_forward = scheme == "forward"
    for i in range(x.size):
        dx = np.zeros_like(x)
        scale = max(1.0, abs(x[i]))
        dx[i] = step * scale
        fp = np.asarray(func(x + dx), dtype=float)
        if use_forward:
            jac[:, i] = (fp - f0) / dx[i]
        else:
            fm = np.asarray(func(x - dx), dtype=float)
            jac[:, i] = (fp - fm) / (2.0 * dx[i])
    return jac


def newton_raphson_fd(
    func,
    x0: np.ndarray,
    tol: float,
    max_iter: int,
    step: float,
    jacobian_scheme: str = "central",
) -> tuple[np.ndarray, int]:
    x = np.asarray(x0, dtype=float).copy()
    residual_cache: dict[bytes, np.ndarray] = {}

    def evaluate(candidate: np.ndarray) -> np.ndarray:
        values = np.asarray(candidate, dtype=float)
        key = values.tobytes()
        cached = residual_cache.get(key)
        if cached is None:
            cached = np.asarray(func(values), dtype=float)
            residual_cache[key] = cached
        return cached.copy()

    for iteration in range(1, max_iter + 1):
        residual = evaluate(x)
        if np.linalg.norm(residual, ord=np.inf) < tol:
            return x, iteration
        jac = finite_difference_jacobian(evaluate, x, step, residual, jacobian_scheme)
        try:
            delta = np.linalg.solve(jac, -residual)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(jac, -residual, rcond=None)
        damping = 1.0
        base_norm = np.linalg.norm(residual, ord=np.inf)
        while damping > 1.0e-3:
            trial = x + damping * delta
            trial_norm = np.linalg.norm(evaluate(trial), ord=np.inf)
            if trial_norm < base_norm:
                x = trial
                break
            damping *= 0.5
        else:
            x = x + delta
    raise NewtonSolveError("Newton-Raphson did not converge within the iteration limit.")
