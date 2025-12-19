import json
import numpy as np


# ==================== Data utilities ====================

def find_datacount(name, filename='log.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        datalog = json.load(f)

    if name in datalog:
        count = datalog[name]
        datalog[name] = count + 1
    else:
        count = 0
        datalog[name] = 1

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(datalog, f, ensure_ascii=False, indent=4)

    return count


# ==================== Orientation utilities ====================

def quaternion_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to axis-angle [rx, ry, rz]."""
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0])
    q_norm = q / norm
    qx, qy, qz, qw = q_norm

    angle = 2 * np.arccos(np.clip(qw, -1.0, 1.0))
    sin_angle = np.sin(angle / 2)

    if sin_angle < 1e-8:
        return np.array([0.0, 0.0, 0.0])

    axis = np.array([qx, qy, qz]) / sin_angle
    return axis * angle


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle [rx, ry, rz] to quaternion [qx, qy, qz, qw]."""
    angle = np.linalg.norm(axis_angle)

    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])

    axis = axis_angle / angle
    half_angle = angle / 2
    sin_half = np.sin(half_angle)

    return np.array([
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        np.cos(half_angle)
    ])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (Hamilton product).

    Args:
        q1: First quaternion [qx, qy, qz, qw]
        q2: Second quaternion [qx, qy, qz, qw]

    Returns:
        Product quaternion q1 * q2 (applies q2 first, then q1)
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / norm
