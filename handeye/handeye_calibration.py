"""
Hand-Eye Calibration using Dual Quaternions

Pure Python implementation of the Daniilidis 1999 method for solving AX = XB.

Reference:
    Daniilidis, K. "Hand-eye calibration using dual quaternions."
    The International Journal of Robotics Research 18.3 (1999): 286-298.

This is a mathematically equivalent reimplementation of the CamOdoCal C++ code,
using only NumPy and SciPy (no ROS, Ceres, or Eigen dependencies).

Usage:
    from handeye_calibration import calibrate_hand_eye

    # robot_poses: list of 4x4 matrices (base -> end-effector)
    # camera_poses: list of 4x4 matrices (camera -> marker/target)
    X = calibrate_hand_eye(robot_poses, camera_poses)
    # X is the 4x4 transform from end-effector to camera
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from typing import List, Tuple, Optional
import warnings


# =============================================================================
# Quaternion Utilities
# =============================================================================

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (wxyz convention).

    Args:
        q1, q2: Quaternions as [w, x, y, z]

    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (wxyz convention)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_exp(q: np.ndarray) -> np.ndarray:
    """
    Quaternion exponential.

    For q = [w, x, y, z], computes exp(q).
    """
    w = q[0]
    v = q[1:4]
    a = np.linalg.norm(v)
    exp_w = np.exp(w)

    if a < 1e-10:
        return np.array([exp_w, 0.0, 0.0, 0.0])

    # sinc(a) = sin(a)/a
    sinc_a = np.sin(a) / a

    result = np.zeros(4)
    result[0] = exp_w * np.cos(a)
    result[1:4] = exp_w * sinc_a * v
    return result


def quat_log(q: np.ndarray) -> np.ndarray:
    """
    Quaternion logarithm.

    For unit quaternion q, computes log(q).
    """
    q_norm = np.linalg.norm(q)
    w = np.log(q_norm)

    # Avoid division by zero
    if q_norm < 1e-10:
        return np.array([0.0, 0.0, 0.0, 0.0])

    # a = acos(q[0] / ||q||)
    cos_a = np.clip(q[0] / q_norm, -1.0, 1.0)
    a = np.arccos(cos_a)

    if a < 1e-10:
        return np.array([w, 0.0, 0.0, 0.0])

    # v / (||q|| * sin(a) / a)
    sin_a_over_a = np.sin(a) / a

    result = np.zeros(4)
    result[0] = w
    result[1:4] = q[1:4] / (q_norm * sin_a_over_a)
    return result


# =============================================================================
# Dual Quaternion Class
# =============================================================================

class DualQuaternion:
    """
    Dual quaternion representation of rigid body transformations.

    A dual quaternion dq = (q_r, q_d) where:
        - q_r: real part (rotation quaternion, wxyz)
        - q_d: dual part (encodes translation)

    For a transform with rotation R and translation t:
        q_r = quaternion(R)
        q_d = 0.5 * quaternion(0, t) * q_r

    Quaternion convention: [w, x, y, z]
    """

    def __init__(self, real: np.ndarray, dual: np.ndarray):
        """
        Initialize dual quaternion from real and dual parts.

        Args:
            real: Real part [w, x, y, z]
            dual: Dual part [w, x, y, z]
        """
        self.real = np.asarray(real, dtype=np.float64)
        self.dual = np.asarray(dual, dtype=np.float64)

    @classmethod
    def from_rotation_translation(cls, rotation: np.ndarray, translation: np.ndarray) -> 'DualQuaternion':
        """
        Create dual quaternion from rotation quaternion and translation vector.

        Args:
            rotation: Rotation quaternion [w, x, y, z] (will be normalized)
            translation: Translation vector [x, y, z]

        Returns:
            DualQuaternion instance
        """
        q_r = rotation / np.linalg.norm(rotation)

        # q_d = 0.5 * t_quat * q_r, where t_quat = [0, tx, ty, tz]
        t_quat = np.array([0.0, translation[0], translation[1], translation[2]])
        q_d = 0.5 * quat_multiply(t_quat, q_r)

        return cls(q_r, q_d)

    @classmethod
    def from_matrix(cls, H: np.ndarray) -> 'DualQuaternion':
        """
        Create dual quaternion from 4x4 homogeneous transformation matrix.

        Args:
            H: 4x4 transformation matrix

        Returns:
            DualQuaternion instance
        """
        R = H[:3, :3]
        t = H[:3, 3]

        # Convert rotation matrix to quaternion (wxyz)
        rot = Rotation.from_matrix(R)
        q_scipy = rot.as_quat()  # scipy uses xyzw
        q_r = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])  # convert to wxyz

        return cls.from_rotation_translation(q_r, t)

    def to_matrix(self) -> np.ndarray:
        """
        Convert dual quaternion to 4x4 homogeneous transformation matrix.

        Returns:
            4x4 transformation matrix
        """
        H = np.eye(4)

        # Rotation from real part
        q_scipy = np.array([self.real[1], self.real[2], self.real[3], self.real[0]])  # wxyz -> xyzw
        rot = Rotation.from_quat(q_scipy)
        H[:3, :3] = rot.as_matrix()

        # Translation: t = 2 * q_d * conj(q_r)
        t_quat = 2.0 * quat_multiply(self.dual, quat_conjugate(self.real))
        H[:3, 3] = t_quat[1:4]

        return H

    def conjugate(self) -> 'DualQuaternion':
        """Return the conjugate of this dual quaternion."""
        return DualQuaternion(quat_conjugate(self.real), quat_conjugate(self.dual))

    def inverse(self) -> 'DualQuaternion':
        """Return the inverse of this dual quaternion."""
        sqr_len0 = np.dot(self.real, self.real)
        sqr_lenE = 2.0 * np.dot(self.real, self.dual)

        if sqr_len0 < 1e-10:
            return DualQuaternion.zeros()

        inv_sqr_len0 = 1.0 / sqr_len0
        inv_sqr_lenE = -sqr_lenE / (sqr_len0 * sqr_len0)

        conj = self.conjugate()
        new_real = inv_sqr_len0 * conj.real
        new_dual = inv_sqr_len0 * conj.dual + inv_sqr_lenE * conj.real

        return DualQuaternion(new_real, new_dual)

    def __mul__(self, other: 'DualQuaternion') -> 'DualQuaternion':
        """Multiply two dual quaternions."""
        new_real = quat_multiply(self.real, other.real)
        new_dual = quat_multiply(self.real, other.dual) + quat_multiply(self.dual, other.real)
        return DualQuaternion(new_real, new_dual)

    def log(self) -> 'DualQuaternion':
        """
        Logarithm of dual quaternion.

        Used in the cost function for refinement.
        """
        real_log = quat_log(self.real)

        sqr_norm = np.dot(self.real, self.real)
        if sqr_norm < 1e-10:
            return DualQuaternion.zeros()

        dual_log = quat_multiply(quat_conjugate(self.real), self.dual) / sqr_norm

        return DualQuaternion(real_log, dual_log)

    def normalize(self) -> 'DualQuaternion':
        """Return normalized dual quaternion."""
        length = np.linalg.norm(self.real)
        if length < 1e-10:
            return DualQuaternion.zeros()

        new_real = self.real / length
        new_dual = self.dual / length

        # Ensure orthogonality: real and dual parts should be orthogonal
        dot_product = np.dot(new_real, new_dual)
        new_dual = new_dual - dot_product * new_real

        return DualQuaternion(new_real, new_dual)

    def squared_norm(self) -> float:
        """Return squared norm of the dual quaternion (real + dual parts)."""
        return np.dot(self.real, self.real) + np.dot(self.dual, self.dual)

    @staticmethod
    def zeros() -> 'DualQuaternion':
        """Return zero dual quaternion."""
        return DualQuaternion(np.zeros(4), np.zeros(4))

    @staticmethod
    def identity() -> 'DualQuaternion':
        """Return identity dual quaternion."""
        return DualQuaternion(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(4))


# =============================================================================
# Screw and Rotation Utilities
# =============================================================================

def skew(v: np.ndarray) -> np.ndarray:
    """
    Create 3x3 skew-symmetric matrix from 3D vector.

    Args:
        v: 3D vector [x, y, z]

    Returns:
        3x3 skew-symmetric matrix
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to axis-angle representation.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Axis-angle vector (axis * angle)
    """
    rot = Rotation.from_matrix(R)
    rotvec = rot.as_rotvec()
    return rotvec


def axis_angle_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle to rotation matrix.

    Args:
        rvec: Axis-angle vector (axis * angle)

    Returns:
        3x3 rotation matrix
    """
    angle = np.linalg.norm(rvec)
    if angle < 1e-10:
        return np.eye(3)

    rot = Rotation.from_rotvec(rvec)
    return rot.as_matrix()


def axis_angle_to_quaternion(rvec: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle to quaternion (wxyz).

    Args:
        rvec: Axis-angle vector (axis * angle)

    Returns:
        Quaternion [w, x, y, z]
    """
    R = axis_angle_to_rotation_matrix(rvec)
    rot = Rotation.from_matrix(R)
    q_scipy = rot.as_quat()  # xyzw
    return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])  # wxyz


def transform_to_screw(rvec: np.ndarray, tvec: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Convert axis-angle rotation and translation to screw parameters.

    Reference: Daniilidis 1999, used in Equations 31, 33.

    Args:
        rvec: Axis-angle rotation vector
        tvec: Translation vector

    Returns:
        Tuple of (theta, d, l, m) where:
            theta: rotation angle
            d: pitch (translation along screw axis)
            l: screw axis direction (unit vector)
            m: screw axis moment
    """
    theta = np.linalg.norm(rvec)

    if theta < 1e-10:
        warnings.warn("Zero rotation in screw conversion, returning zeros")
        return 0.0, 0.0, np.zeros(3), np.zeros(3)

    # Screw axis direction (unit vector)
    l = rvec / theta

    # Pitch: projection of translation onto screw axis
    d = np.dot(tvec, l)

    # Point on screw axis
    # c = 0.5 * (t - d*l + cot(theta/2) * (l x t))
    cot_half_theta = 1.0 / np.tan(theta / 2.0)
    c = 0.5 * (tvec - d * l + cot_half_theta * np.cross(l, tvec))

    # Screw axis moment
    m = np.cross(c, l)

    return theta, d, l, m


# =============================================================================
# SVD-based Initial Estimate (Daniilidis 1999)
# =============================================================================

def build_S_transpose_block(a: np.ndarray, a_prime: np.ndarray,
                            b: np.ndarray, b_prime: np.ndarray) -> np.ndarray:
    """
    Build a 6x8 S^T block for the T matrix.

    Reference: Daniilidis 1999, Equations 31, 33.

    Args:
        a, a_prime: Screw parameters from first transform (l, m)
        b, b_prime: Screw parameters from second transform (l, m)

    Returns:
        6x8 matrix block
    """
    S_T = np.zeros((6, 8))

    skew_a_plus_b = skew(a + b)
    a_minus_b = a - b

    # First 3 rows
    S_T[0:3, 0] = a_minus_b
    S_T[0:3, 1:4] = skew_a_plus_b

    # Last 3 rows
    S_T[3:6, 0] = a_prime - b_prime
    S_T[3:6, 1:4] = skew(a_prime + b_prime)
    S_T[3:6, 4] = a_minus_b
    S_T[3:6, 5:8] = skew_a_plus_b

    return S_T


def build_T_matrix(rvecs1: List[np.ndarray], tvecs1: List[np.ndarray],
                   rvecs2: List[np.ndarray], tvecs2: List[np.ndarray]) -> np.ndarray:
    """
    Build the T matrix for SVD from motion pairs.

    Args:
        rvecs1, tvecs1: Axis-angle rotations and translations for first set
        rvecs2, tvecs2: Axis-angle rotations and translations for second set

    Returns:
        (N*6) x 8 matrix T
    """
    n = len(rvecs1)
    T = np.zeros((n * 6, 8))

    for i in range(n):
        rvec1, tvec1 = rvecs1[i], tvecs1[i]
        rvec2, tvec2 = rvecs2[i], tvecs2[i]

        # Skip zero rotations
        if np.linalg.norm(rvec1) < 1e-10 or np.linalg.norm(rvec2) < 1e-10:
            continue

        # Convert to screw parameters
        _, _, l1, m1 = transform_to_screw(rvec1, tvec1)
        _, _, l2, m2 = transform_to_screw(rvec2, tvec2)

        # Build block
        T[i*6:(i+1)*6, :] = build_S_transpose_block(l1, m1, l2, m2)

    return T


def solve_quadratic(a: float, b: float, c: float) -> Tuple[float, float]:
    """
    Solve quadratic equation ax^2 + bx + c = 0.

    Returns:
        Tuple of two solutions (x1, x2)
    """
    delta2 = b * b - 4.0 * a * c

    if delta2 < 0:
        # No real solutions, return zeros
        return 0.0, 0.0

    delta = np.sqrt(delta2)
    x1 = (-b + delta) / (2.0 * a)
    x2 = (-b - delta) / (2.0 * a)

    return x1, x2


def estimate_initial_dual_quaternion(T: np.ndarray, planar_motion: bool = False) -> DualQuaternion:
    """
    Estimate initial dual quaternion using SVD.

    Reference: Daniilidis 1999, Section 6.

    Args:
        T: The (N*6) x 8 matrix from build_T_matrix
        planar_motion: If True, handle rank-5 case for planar motion

    Returns:
        Initial estimate as DualQuaternion
    """
    # SVD
    U, s, Vt = np.linalg.svd(T, full_matrices=True)
    V = Vt.T

    # Solution lies in null space (last two columns of V)
    v6 = V[:, 5]
    v7 = V[:, 6]
    v8 = V[:, 7]

    # Handle planar motion (rank 5)
    if planar_motion:
        v7 = v7 + v6

    # Extract components
    u1 = v7[0:4]
    v1 = v7[4:8]
    u2 = v8[0:4]
    v2 = v8[4:8]

    # Find lambda1, lambda2 such that:
    # q = lambda1 * v7 + lambda2 * v8
    # with constraints from unit dual quaternion

    if np.abs(np.dot(u1, v1)) < 1e-10:
        u1, u2 = u2, u1
        v1, v2 = v2, v1

    if np.abs(np.dot(u1, v1)) > 1e-10:
        # Solve quadratic for s = lambda1 / lambda2
        a_coef = np.dot(u1, v1)
        b_coef = np.dot(u1, v2) + np.dot(u2, v1)
        c_coef = np.dot(u2, v2)

        s1, s2 = solve_quadratic(a_coef, b_coef, c_coef)

        # Find better solution
        t1 = s1 * s1 * np.dot(u1, u1) + 2 * s1 * np.dot(u1, u2) + np.dot(u2, u2)
        t2 = s2 * s2 * np.dot(u1, u1) + 2 * s2 * np.dot(u1, u2) + np.dot(u2, u2)

        if t1 > t2:
            s = s1
            t = t1
        else:
            s = s2
            t = t2

        if t < 1e-10:
            raise ValueError("Cannot normalize - check your input transforms")

        lambda2 = np.sqrt(1.0 / t)
        lambda1 = s * lambda2
    else:
        # Handle special cases
        u1_norm = np.linalg.norm(u1)
        u2_norm = np.linalg.norm(u2)

        if u1_norm < 1e-10 and u2_norm > 1e-10:
            lambda1 = 0.0
            lambda2 = 1.0 / u2_norm
        elif u2_norm < 1e-10 and u1_norm > 1e-10:
            lambda1 = 1.0 / u1_norm
            lambda2 = 0.0
        else:
            raise ValueError(
                "Cannot normalize dual quaternion. "
                "Check that your transforms are properly aligned and have consistent rotations."
            )

    # Construct dual quaternion
    q_coeffs = lambda1 * u1 + lambda2 * u2
    q_prime_coeffs = lambda1 * v1 + lambda2 * v2

    # Create dual quaternion (coefficients are in wxyz order)
    q_real = np.array([q_coeffs[0], q_coeffs[1], q_coeffs[2], q_coeffs[3]])
    q_dual = np.array([q_prime_coeffs[0], q_prime_coeffs[1], q_prime_coeffs[2], q_prime_coeffs[3]])

    return DualQuaternion(q_real, q_dual)


# =============================================================================
# Non-linear Refinement
# =============================================================================

def compute_pose_error(dq: DualQuaternion,
                       rvec1: np.ndarray, tvec1: np.ndarray,
                       rvec2: np.ndarray, tvec2: np.ndarray) -> float:
    """
    Compute pose error for a single motion pair.

    Error = ||log(dq1^{-1} * dq * dq2 * dq^{-1})||^2

    This measures how well AX = XB is satisfied.
    """
    # Create dual quaternions for the motions
    q1 = axis_angle_to_quaternion(rvec1)
    q2 = axis_angle_to_quaternion(rvec2)

    dq1 = DualQuaternion.from_rotation_translation(q1, tvec1)
    dq2 = DualQuaternion.from_rotation_translation(q2, tvec2)

    # dq1_ = dq * dq2 * dq^{-1}
    dq_inv = dq.inverse()
    dq1_prime = dq * dq2 * dq_inv

    # diff = log(dq1^{-1} * dq1_)
    diff = (dq1.inverse() * dq1_prime).log()

    # Error = ||real||^2 + ||dual||^2
    error = np.dot(diff.real, diff.real) + np.dot(diff.dual, diff.dual)

    return error


def cost_function(params: np.ndarray,
                  rvecs1: List[np.ndarray], tvecs1: List[np.ndarray],
                  rvecs2: List[np.ndarray], tvecs2: List[np.ndarray]) -> float:
    """
    Total cost function for optimization.

    Args:
        params: [qw, qx, qy, qz, tx, ty, tz] - quaternion (wxyz) and translation
        rvecs1, tvecs1: First set of motions
        rvecs2, tvecs2: Second set of motions

    Returns:
        Total error
    """
    # Extract and normalize quaternion
    q = params[0:4]
    q = q / np.linalg.norm(q)
    t = params[4:7]

    dq = DualQuaternion.from_rotation_translation(q, t)

    total_error = 0.0
    for i in range(len(rvecs1)):
        error = compute_pose_error(dq, rvecs1[i], tvecs1[i], rvecs2[i], tvecs2[i])
        total_error += error

    return total_error


def refine_estimate(dq_init: DualQuaternion,
                    rvecs1: List[np.ndarray], tvecs1: List[np.ndarray],
                    rvecs2: List[np.ndarray], tvecs2: List[np.ndarray],
                    verbose: bool = True,
                    max_iterations: int = 500,
                    num_restarts: int = 1) -> DualQuaternion:
    """
    Refine initial dual quaternion estimate using non-linear optimization.

    Args:
        dq_init: Initial dual quaternion estimate
        rvecs1, tvecs1: First set of motions
        rvecs2, tvecs2: Second set of motions
        verbose: Print optimization info
        max_iterations: Maximum iterations per optimization run
        num_restarts: Number of optimization restarts with perturbation

    Returns:
        Refined DualQuaternion
    """
    # Extract initial parameters
    H_init = dq_init.to_matrix()
    q_init = dq_init.real
    t_init = H_init[:3, 3]

    # Initial parameter vector [qw, qx, qy, qz, tx, ty, tz]
    x0 = np.concatenate([q_init, t_init])

    initial_cost = cost_function(x0, rvecs1, tvecs1, rvecs2, tvecs2)
    if verbose:
        print(f"Initial cost: {initial_cost:.6e}")

    best_result = None
    best_cost = float('inf')

    for restart in range(num_restarts):
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb the initial estimate for restarts
            x_start = x0.copy()
            x_start[:4] += np.random.randn(4) * 0.01  # Small quaternion perturbation
            x_start[:4] /= np.linalg.norm(x_start[:4])  # Re-normalize
            x_start[4:] += np.random.randn(3) * 0.005  # Small translation perturbation (5mm)

        if verbose and num_restarts > 1:
            print(f"  Optimization restart {restart + 1}/{num_restarts}...")

        # First pass: BFGS
        result = minimize(
            cost_function,
            x_start,
            args=(rvecs1, tvecs1, rvecs2, tvecs2),
            method='BFGS',
            options={
                'maxiter': max_iterations,
                'disp': verbose and num_restarts == 1,
                'gtol': 1e-8,
            }
        )

        # Second pass: L-BFGS-B for tighter convergence
        result = minimize(
            cost_function,
            result.x,
            args=(rvecs1, tvecs1, rvecs2, tvecs2),
            method='L-BFGS-B',
            options={
                'maxiter': max_iterations,
                'disp': verbose and num_restarts == 1,
                'ftol': 1e-12,
                'gtol': 1e-10,
            }
        )

        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result

    if verbose:
        print(f"Optimization complete: initial cost = {initial_cost:.6e}, "
              f"final cost = {best_cost:.6e}, "
              f"converged = {best_result.success}, "
              f"iterations = {best_result.nit}")

    # Extract result
    q_opt = best_result.x[0:4]
    q_opt = q_opt / np.linalg.norm(q_opt)
    t_opt = best_result.x[4:7]

    return DualQuaternion.from_rotation_translation(q_opt, t_opt)


# =============================================================================
# Main API
# =============================================================================

def calibrate_hand_eye(
    robot_poses: List[np.ndarray],
    camera_poses: List[np.ndarray],
    planar_motion: bool = False,
    verbose: bool = True,
    max_iterations: int = 500,
    num_restarts: int = 1
) -> np.ndarray:
    """
    Perform hand-eye calibration using dual quaternion method.

    Solves the AX = XB problem where:
        - A: robot end-effector motions (between poses)
        - B: camera/marker motions (between poses)
        - X: unknown transform from end-effector to camera

    Args:
        robot_poses: List of 4x4 matrices representing robot end-effector poses
                     (base frame -> end-effector frame)
        camera_poses: List of 4x4 matrices representing camera/marker poses
                      (camera frame -> marker/target frame)
        planar_motion: Set True if robot moves only in a plane (rank-5 case)
        verbose: Print progress information
        max_iterations: Maximum iterations per optimization run (default 500)
        num_restarts: Number of optimization restarts with perturbation (default 1)

    Returns:
        4x4 homogeneous transformation matrix X (end-effector -> camera)

    Example:
        >>> robot_poses = [T_base_ee_1, T_base_ee_2, ...]  # 4x4 matrices
        >>> camera_poses = [T_cam_marker_1, T_cam_marker_2, ...]  # 4x4 matrices
        >>> X = calibrate_hand_eye(robot_poses, camera_poses)
        >>> # X transforms from end-effector frame to camera frame

    Notes:
        - Requires at least 3 pose pairs (more is better, ~20-40 recommended)
        - Poses should cover diverse orientations for best results
        - Avoid purely translational motions (need rotational component)
    """
    if len(robot_poses) != len(camera_poses):
        raise ValueError("Number of robot poses must match number of camera poses")

    if len(robot_poses) < 3:
        raise ValueError("At least 3 pose pairs are required for calibration")

    if verbose:
        print(f"Hand-eye calibration with {len(robot_poses)} pose pairs")

    # Convert absolute poses to relative motions (relative to first pose)
    #
    # For eye-in-hand with fixed marker:
    #   T_base_marker = T_base_ee @ X @ T_cam_marker = constant
    #
    # This gives the constraint: A @ X = X @ B where:
    #   A = inv(T_base_ee[0]) @ T_base_ee[i]  (relative robot motion)
    #   B = T_cam_marker[0] @ inv(T_cam_marker[i])  (relative camera observation)
    #
    first_robot_inv = np.linalg.inv(robot_poses[0])
    first_camera = camera_poses[0]

    rvecs_robot = []
    tvecs_robot = []
    rvecs_camera = []
    tvecs_camera = []

    for i in range(1, len(robot_poses)):
        # A = inv(T_base_ee[0]) @ T_base_ee[i]
        robot_rel = first_robot_inv @ robot_poses[i]

        # B = T_cam_marker[0] @ inv(T_cam_marker[i])
        camera_rel = first_camera @ np.linalg.inv(camera_poses[i])

        # Extract axis-angle and translation
        rvec_robot = rotation_matrix_to_axis_angle(robot_rel[:3, :3])
        tvec_robot = robot_rel[:3, 3]

        rvec_camera = rotation_matrix_to_axis_angle(camera_rel[:3, :3])
        tvec_camera = camera_rel[:3, 3]

        # Skip if rotation is too small
        if np.linalg.norm(rvec_robot) < 1e-6 or np.linalg.norm(rvec_camera) < 1e-6:
            if verbose:
                print(f"  Skipping pair {i}: insufficient rotation")
            continue

        rvecs_robot.append(rvec_robot)
        tvecs_robot.append(tvec_robot)
        rvecs_camera.append(rvec_camera)
        tvecs_camera.append(tvec_camera)

    if len(rvecs_robot) < 2:
        raise ValueError("Not enough valid motion pairs after filtering")

    if verbose:
        print(f"  Using {len(rvecs_robot)} valid motion pairs")

    # Build T matrix and get initial estimate via SVD
    T = build_T_matrix(rvecs_robot, tvecs_robot, rvecs_camera, tvecs_camera)
    dq_init = estimate_initial_dual_quaternion(T, planar_motion)

    H_init = dq_init.to_matrix()
    if verbose:
        print("Initial estimate (before refinement):")
        print(H_init)

    # Refine using non-linear optimization
    dq_refined = refine_estimate(
        dq_init,
        rvecs_robot, tvecs_robot,
        rvecs_camera, tvecs_camera,
        verbose=verbose,
        max_iterations=max_iterations,
        num_restarts=num_restarts
    )

    H_refined = dq_refined.to_matrix()
    if verbose:
        print("Refined estimate (after optimization):")
        print(H_refined)

        # Print translation and rotation
        t = H_refined[:3, 3]
        rot = Rotation.from_matrix(H_refined[:3, :3])
        q = rot.as_quat()  # xyzw
        print(f"Translation (x,y,z): {t}")
        print(f"Rotation quaternion (w,x,y,z): {q[3]:.6f}, {q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}")

    return H_refined


# =============================================================================
# File I/O Utilities
# =============================================================================

def load_transform_pairs_yaml(filename: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load transform pairs from YAML file (OpenCV format).

    Compatible with the reference implementation's TransformPairsInput.yml format.

    Args:
        filename: Path to YAML file

    Returns:
        Tuple of (robot_poses, camera_poses) as lists of 4x4 matrices
    """
    import yaml

    # OpenCV YAML has a special format, need custom parsing
    with open(filename, 'r') as f:
        content = f.read()

    # Remove OpenCV-specific tags
    content = content.replace('!!opencv-matrix', '')

    # Parse YAML
    data = yaml.safe_load(content)

    frame_count = data['frameCount']
    robot_poses = []
    camera_poses = []

    for i in range(frame_count):
        # Parse T1 (robot)
        t1_data = data[f'T1_{i}']['data']
        T1 = np.array(t1_data).reshape(4, 4)
        robot_poses.append(T1)

        # Parse T2 (camera)
        t2_data = data[f'T2_{i}']['data']
        T2 = np.array(t2_data).reshape(4, 4)
        camera_poses.append(T2)

    return robot_poses, camera_poses


def save_calibration_result(H: np.ndarray, filename: str):
    """
    Save calibration result to file.

    Args:
        H: 4x4 transformation matrix
        filename: Output filename (.npy or .txt)
    """
    if filename.endswith('.npy'):
        np.save(filename, H)
    else:
        np.savetxt(filename, H, fmt='%.10f')
    print(f"Saved calibration result to {filename}")


# =============================================================================
# Command-line interface
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Hand-eye calibration using dual quaternions')
    parser.add_argument('input_file', help='Input YAML file with transform pairs')
    parser.add_argument('-o', '--output', help='Output file for calibration result')
    parser.add_argument('--planar', action='store_true', help='Use planar motion mode')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    # Load data
    robot_poses, camera_poses = load_transform_pairs_yaml(args.input_file)

    # Calibrate
    X = calibrate_hand_eye(
        robot_poses,
        camera_poses,
        planar_motion=args.planar,
        verbose=not args.quiet
    )

    # Save if requested
    if args.output:
        save_calibration_result(X, args.output)
