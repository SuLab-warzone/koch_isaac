"""Small estimator contract shared by simulation and future real-camera adapters."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class BoxEstimate:
    # Environment/robot reference frame, metres. Never consume position when invalid.
    position: np.ndarray
    valid: bool
    reason: str
    pixel_uv: tuple[float, float] | None = None
    area_px: int = 0

    @classmethod
    def missing(cls, reason, pixel_uv=None, area_px=0):
        return cls(np.zeros(3, dtype=np.float32), False, reason, pixel_uv, area_px)

    def as_dict(self):
        return {"position_env_m": self.position.tolist() if self.valid else None,
                "valid": self.valid, "reason": self.reason,
                "pixel_uv": self.pixel_uv, "area_px": self.area_px}


class BoxEstimator(Protocol):
    """Future depth/learned estimators implement this same interface.

    Input is uint8 RGB, NOT OpenCV's usual BGR. Output must use the same metre
    coordinate frame as robot forward kinematics. Camera calibration is supplied
    when constructing the estimator, never inferred from simulator object poses.
    """
    def estimate(self, rgb: np.ndarray) -> BoxEstimate: ...
