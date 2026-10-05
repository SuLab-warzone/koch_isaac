"""Image-to-horizontal-plane calibration, independent of Isaac and robot control."""
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PlaneCalibration:
    pixel_to_xy: np.ndarray
    image_size: tuple[int, int]  # width, height; resizing silently invalidates calibration
    plane_z_m: float            # visible surface height used for image projection
    center_z_m: float           # assumed box centre height while resting on the table

    def __post_init__(self):
        self.pixel_to_xy = np.asarray(self.pixel_to_xy, dtype=np.float64)
        self.image_size = tuple(self.image_size)
        if (self.pixel_to_xy.shape != (3, 3) or
                not np.isfinite(self.pixel_to_xy).all() or
                np.linalg.matrix_rank(self.pixel_to_xy) != 3):
            raise ValueError("pixel_to_xy must be a finite, invertible 3x3 matrix")
        if len(self.image_size) != 2 or any(int(n) != n or n <= 0 for n in self.image_size):
            raise ValueError("image_size must be [width, height] in pixels")
        if not np.isfinite([self.plane_z_m, self.center_z_m]).all():
            raise ValueError("Plane/centre heights must be finite")

    @classmethod
    def from_camera(cls, intrinsic, rotation_env_from_camera, camera_position_env,
                    image_size, plane_z_m, center_z_m):
        """Use a pinhole camera: optical +X right, +Y down, +Z forward (ROS).

        A plane point [x,y,z_plane] projects as K R_camera_from_env (point-camera).
        Inverting that 3x3 plane projection gives pixels -> environment XY.
        Isaac supplies camera calibration here; no box pose is used.
        """
        k = np.asarray(intrinsic, dtype=np.float64)
        r = np.asarray(rotation_env_from_camera, dtype=np.float64).T
        p = np.asarray(camera_position_env, dtype=np.float64)
        projection = k @ np.column_stack(
            (r[:, 0], r[:, 1], r @ (np.array([0., 0., plane_z_m]) - p)))
        return cls(np.linalg.inv(projection), image_size, plane_z_m, center_z_m)

    @classmethod
    def from_points(cls, pixels_uv, plane_xy_m, image_size, plane_z_m, center_z_m):
        """Fit >=4 measured point pairs. These are calibration, not training labels.

        Use undistorted pixel coordinates and points spanning the grasp workspace
        at the specified plane height. Verify on held-out points afterwards.
        """
        pixels, xy = np.asarray(pixels_uv, float), np.asarray(plane_xy_m, float)
        if (pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.shape != xy.shape
                or len(pixels) < 4 or not np.isfinite(pixels).all() or not np.isfinite(xy).all()):
            raise ValueError("Supply at least four finite matching [u,v] and [x,y] pairs")
        for points in (pixels, xy):
            if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
                raise ValueError("Calibration points must not be collinear")
        h, _ = cv2.findHomography(pixels, xy, method=0)
        if h is None:
            raise ValueError("Cannot fit plane calibration")
        return cls(h, image_size, plane_z_m, center_z_m)

    def project(self, uv):
        homogeneous = self.pixel_to_xy @ np.array([*uv, 1.], dtype=np.float64)
        if not np.isfinite(homogeneous).all() or abs(homogeneous[2]) < 1e-10:
            return None
        xy = homogeneous[:2] / homogeneous[2]
        return np.array([*xy, self.center_z_m], dtype=np.float32)

    def as_dict(self):
        return {"pixel_to_xy": self.pixel_to_xy.tolist(), "image_size": list(self.image_size),
                "plane_z_m": self.plane_z_m, "center_z_m": self.center_z_m}

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
