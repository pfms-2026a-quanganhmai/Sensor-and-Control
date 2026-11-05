"""模型、可控性/可观性与 LQR 数值证据；不把局部线性结论推广为全局结论。"""

import json
from pathlib import Path

import numpy as np

from .control import DT, linear_model, lqr_gain


def analyse():
    a, b = linear_model()
    c = np.eye(3)
    k = lqr_gain()
    controllability = np.column_stack([b, a @ b, a @ a @ b])
    observability = np.vstack([c, c @ a, c @ a @ a])
    stopped_a, stopped_b = linear_model(0.0)
    stopped = np.column_stack(
        [stopped_b, stopped_a @ stopped_b, stopped_a @ stopped_a @ stopped_b]
    )
    eigenvalues = np.linalg.eigvals(a - b @ k)
    return {
        "state": ["along_track_error_m", "cross_track_error_m", "heading_error_rad"],
        "input": ["delta_linear_m_s", "delta_angular_rad_s"],
        "reference_speed_m_s": 0.12,
        "dt_s": DT,
        "A": a.tolist(),
        "B": b.tolist(),
        "C_camera": c.tolist(),
        "controllability_rank": int(np.linalg.matrix_rank(controllability)),
        "camera_observability_rank": int(np.linalg.matrix_rank(observability)),
        "stationary_linearisation_controllability_rank": int(
            np.linalg.matrix_rank(stopped)
        ),
        "LQR_Q": [6, 40, 10],
        "LQR_R": [4, 2],
        "LQR_K": k.tolist(),
        "closed_loop_eigenvalues": eigenvalues.real.tolist(),
        "closed_loop_spectral_radius": float(max(abs(eigenvalues))),
        "limits": [
            "local moving-reference error model, not global nonlinear controllability proof",
            "camera visible and calibrated: H=I; encoders alone do not observe absolute pose drift",
            "in-place alignment and docking use separate nonlinear state feedback",
            "command saturation, slip and lost measurements require experimental validation",
        ],
    }


def save(output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(analyse(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
