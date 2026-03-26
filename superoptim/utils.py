import os
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
import time 

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

# ---------------------------------------------------------------------------
# Full SDF with tapering & bending  (numpy mirror of BatchSuperQMulti.sdf_batch)
# ---------------------------------------------------------------------------

def _safe_pow_np(x, y, eps=1e-3):
    """Numerically safe power: abs-clamp then power."""
    safe_x = np.clip(np.abs(x), eps, 5e2)
    return np.clip(np.power(safe_x, y), eps, 5e2)

def _safe_mul_np(x, y, eps=1e-4):
    """Numerically safe sign-preserving multiply."""
    xs = np.sign(x) * np.clip(np.abs(x), eps, 1e6)
    ys = np.sign(y) * np.clip(np.abs(y), eps, 1e6)
    return xs * ys + eps

def _inverse_bending_axis_np(x, y, z, kb, alpha, axis):
    """Inverse bending deformation along a single axis (numpy version)."""
    if axis == 'z':
        u, v, w = x, y, z
    elif axis == 'x':
        u, v, w = y, z, x
    elif axis == 'y':
        u, v, w = z, x, y
    else:
        return x, y, z

    inv_kb = 1.0 / (kb + 1e-6)
    angle_offset = np.arctan2(v, u)
    R = np.sqrt(u**2 + v**2) * np.cos(alpha - angle_offset)
    gamma = np.arctan2(w, inv_kb - R)
    r = inv_kb - np.sqrt(w**2 + (inv_kb - R)**2)

    u = u - (R - r) * np.cos(alpha)
    v = v - (R - r) * np.sin(alpha)
    w = inv_kb * gamma

    if axis == 'z':
        return u, v, w
    elif axis == 'x':
        return w, u, v
    elif axis == 'y':
        return v, w, u

def sdf_superquadric_full(points, scale_vec, exponents, translation, rotation_matrix,
                           tapering=None, bending=None, truncation=1):
    """
    Full Superquadric SDF supporting tapering and bending deformations.
    Mirrors BatchSuperQMulti.sdf_batch logic in numpy.

    Args:
        points:           (3, N) query points.
        scale_vec:        (3,)  [sx, sy, sz].
        exponents:        (2,)  [e1, e2].
        translation:      (3,)  [tx, ty, tz].
        rotation_matrix:  (3, 3) rotation matrix.
        tapering:         (2,)  [kx, ky] or None.
        bending:          (6,)  [kb_z, a_z, kb_x, a_x, kb_y, a_y] or None.
        truncation:       float, clip distance (0 = disabled).
    """
    eps = 1e-6

    # Transform to local frame
    X = rotation_matrix.T @ (points - translation[:, np.newaxis])

    e1, e2 = exponents
    sx, sy, sz = scale_vec

    x = X[0]
    y = X[1]
    z = X[2]

    # Sign-preserving epsilon clamp (mirrors torch.where(>0,1,-1)*clamp)
    x = np.where(x > 0, 1.0, -1.0) * np.clip(np.abs(x), eps, None)
    y = np.where(y > 0, 1.0, -1.0) * np.clip(np.abs(y), eps, None)
    z = np.where(z > 0, 1.0, -1.0) * np.clip(np.abs(z), eps, None)

    # Inverse bending
    if bending is not None:
        kb_z, a_z, kb_x, a_x, kb_y, a_y = bending
        if kb_z > 1e-6:
            x, y, z = _inverse_bending_axis_np(x, y, z, kb_z, a_z, 'z')
        if kb_x > 1e-6:
            x, y, z = _inverse_bending_axis_np(x, y, z, kb_x, a_x, 'x')
        if kb_y > 1e-6:
            x, y, z = _inverse_bending_axis_np(x, y, z, kb_y, a_y, 'y')

    # Inverse tapering
    if tapering is not None:
        kx, ky = tapering
        fx = kx / sz * z + 1
        fy = ky / sz * z + 1
        fx = np.where(fx > 0, 1.0, -1.0) * np.clip(np.abs(fx), eps, None)
        fy = np.where(fy > 0, 1.0, -1.0) * np.clip(np.abs(fy), eps, None)
        x = x / fx
        y = y / fy

    r0 = np.sqrt(x**2 + y**2 + z**2)

    term1 = _safe_pow_np(_safe_pow_np(x / sx, 2), 1 / e2)
    term2 = _safe_pow_np(_safe_pow_np(y / sy, 2), 1 / e2)
    term3 = _safe_pow_np(_safe_pow_np(z / sz, 2), 1 / e1)

    f_func = _safe_pow_np(_safe_pow_np(term1 + term2, e2 / e1) + term3, -e1 / 2)
    sdf = _safe_mul_np(r0, 1.0 - f_func)
    
    # Enforce proper distance growth outside the primitive:
    # SDF should be at least r0 - radius_approx (linear growth outside bounding sphere)
    # This prevents tiny primitives from generating incorrect near-zero SDF values
    radius = np.sqrt(sx**2 + sy**2 + sz**2)
    floor = r0 - radius
    sdf = np.where(floor > sdf, floor, sdf)

    if truncation != 0:
        sdf = np.clip(sdf, -truncation, truncation)
    
    return sdf

def sdf_union_pred_handler(pred_handler, idx, points, truncation=1.0):
    """
    Compute the union SDF (min over all active primitives) for one entry in
    a PredictionHandler using the full model (tapering + bending).

    Args:
        pred_handler: PredictionHandler instance.
        idx:          Batch index.
        points:       (3, N) or (N, 3) array of query points.
        truncation:   Clip distance.
    Returns:
        sdf: (N,) array of union SDF values.
    """
    if points.shape[0] != 3:
        points = points.T  # ensure (3, N)

    mask = (pred_handler.exist[idx] > 0.5).reshape(-1)
    sqscale    = np.array(pred_handler.scale[idx].reshape(-1, 3)[mask])
    exponents  = np.array(pred_handler.exponents[idx].reshape(-1, 2)[mask])
    translation = np.array(pred_handler.translation[idx].reshape(-1, 3)[mask])
    rotation   = np.array(pred_handler.rotation[idx].reshape(-1, 3, 3)[mask])
    taper      = np.array(pred_handler.tapering[idx].reshape(-1, 2)[mask])
    bending    = np.array(pred_handler.bending[idx].reshape(-1, 6)[mask])

    N = points.shape[1]
    union_sdf = np.full(N, np.inf)
    for i in range(len(sqscale)):
        s = sdf_superquadric_full(
            points, sqscale[i], exponents[i], translation[i], rotation[i],
            tapering=taper[i], bending=bending[i], truncation=truncation,
        )
        union_sdf = np.minimum(union_sdf, s)
    return union_sdf

def sdf_slice_to_image(pred_handler, idx, fixed_axis, fixed_val, limit, res=200, grid_range=0.75, return_data=False):
    """
    Render an SDF slice into an in-memory (H, W, 3) uint8 RGB numpy array.

    Args:
        pred_handler: PredictionHandler instance.
        idx:          Batch index.
        fixed_axis:   'x', 'y', or 'z' — which axis to hold constant.
        fixed_val:    World coordinate of the fixed axis.
        limit:        SDF truncation / colormap range.
        res:          Grid resolution (pixels per side).
        grid_range:   Spatial extent on each free axis.
    Returns:
        img: (H, W, 3) uint8 RGB array.
        If `return_data` is True, also returns `(sdf_grid, u_range, v_range)`.
    """
    u_range = np.linspace(-grid_range, grid_range, res)
    v_range = np.linspace(-grid_range, grid_range, res)
    U_grid, V_grid = np.meshgrid(u_range, v_range)
    fixed = np.full_like(U_grid, fixed_val)

    if fixed_axis == 'x':
        points = np.stack([fixed.ravel(), U_grid.ravel(), V_grid.ravel()], axis=0)
    elif fixed_axis == 'y':
        points = np.stack([U_grid.ravel(), fixed.ravel(), V_grid.ravel()], axis=0)
    else:  # 'z'
        points = np.stack([U_grid.ravel(), V_grid.ravel(), fixed.ravel()], axis=0)

    sdf_vals = sdf_union_pred_handler(pred_handler, idx, points, truncation=limit)
    sdf_grid = sdf_vals.reshape(res, res)

    # Map [-limit, limit] -> [0, 1] then apply RdBu colormap
    t = np.clip(sdf_grid / limit, -1.0, 1.0) * 0.5 + 0.5   # [0,1], 0.5 = surface
    cmap = plt.get_cmap('RdBu')
    img = (cmap(t)[:, :, :3] * 255).astype(np.uint8)        # (H,W,3)

    # Draw zero-contour in black (simple threshold on adjacent signs)
    # from skimage import measure  # type: ignore
    # try:
    #     contours = measure.find_contours(sdf_grid, 0.0)
    #     for contour in contours:
    #         rows = np.clip(contour[:, 0].astype(int), 0, res - 1)
    #         cols = np.clip(contour[:, 1].astype(int), 0, res - 1)
    #         img[rows, cols] = [0, 0, 0]
    # except Exception:
    #     pass

    if return_data:
        return img, sdf_grid, u_range, v_range
    return img

def plot_pred_handler(pred_handler, truncation, wolrd_y=0.0, idx=0, filename="superq_plot.png"):
    # Render SDF slice using the shared utility and save image to file.
        # Use sdf_slice_to_image to compute the SDF grid and ranges, then
        # render using matplotlib so we can add axes and a colorbar.
        img, sdf_grid, u_range, v_range = sdf_slice_to_image(
            pred_handler, idx, fixed_axis='y', fixed_val=wolrd_y, limit=truncation, return_data=True
        )

        plt.figure(figsize=(6, 5))
        extent = [u_range[0], u_range[-1], v_range[0], v_range[-1]]
        plt.imshow(sdf_grid, cmap='RdBu', origin='lower', extent=extent, vmin=-truncation, vmax=truncation)
        plt.colorbar(label='Signed Distance')
        plt.xlabel('X (World)')
        plt.ylabel('Z (World)')
        plt.title(f'Superquadric SDF Slice at Y={wolrd_y}')
        # Draw zero contour
        try:
            Xg, Zg = np.meshgrid(u_range, v_range)
            plt.contour(Xg, Zg, sdf_grid, levels=[0], colors='black', linewidths=1)
        except Exception:
            pass

        plt.savefig(filename)
        plt.close()
        print(f"Plot saved as {filename}")

# https://behavior.stanford.edu/reference/utils/transform_utils.html#utils.transform_utils.mat2quat
def mat2quat(rmat: torch.Tensor) -> torch.Tensor:
    """
    Converts given rotation matrix to quaternion.
    Args:
        rmat (torch.Tensor): (3, 3) or (..., 3, 3) rotation matrix
    Returns:
        torch.Tensor: (4,) or (..., 4) (x,y,z,w) float quaternion angles
    """
    assert torch.allclose(torch.linalg.det(rmat), torch.tensor(1.0)), "Rotation matrix must not be scaled"

    # Check if input is a single matrix or a batch
    is_single = rmat.dim() == 2
    if is_single:
        rmat = rmat.unsqueeze(0)

    batch_shape = rmat.shape[:-2]
    mat_flat = rmat.reshape(-1, 3, 3)

    m00, m01, m02 = mat_flat[:, 0, 0], mat_flat[:, 0, 1], mat_flat[:, 0, 2]
    m10, m11, m12 = mat_flat[:, 1, 0], mat_flat[:, 1, 1], mat_flat[:, 1, 2]
    m20, m21, m22 = mat_flat[:, 2, 0], mat_flat[:, 2, 1], mat_flat[:, 2, 2]

    trace = m00 + m11 + m22

    trace_positive = trace > 0
    cond1 = (m00 > m11) & (m00 > m22) & ~trace_positive
    cond2 = (m11 > m22) & ~(trace_positive | cond1)
    cond3 = ~(trace_positive | cond1 | cond2)

    # Trace positive condition
    sq = torch.where(trace_positive, torch.sqrt(trace + 1.0) * 2.0, torch.zeros_like(trace))
    qw = torch.where(trace_positive, 0.25 * sq, torch.zeros_like(trace))
    qx = torch.where(trace_positive, (m21 - m12) / sq, torch.zeros_like(trace))
    qy = torch.where(trace_positive, (m02 - m20) / sq, torch.zeros_like(trace))
    qz = torch.where(trace_positive, (m10 - m01) / sq, torch.zeros_like(trace))

    # Condition 1
    sq = torch.where(cond1, torch.sqrt(1.0 + m00 - m11 - m22) * 2.0, sq)
    qw = torch.where(cond1, (m21 - m12) / sq, qw)
    qx = torch.where(cond1, 0.25 * sq, qx)
    qy = torch.where(cond1, (m01 + m10) / sq, qy)
    qz = torch.where(cond1, (m02 + m20) / sq, qz)

    # Condition 2
    sq = torch.where(cond2, torch.sqrt(1.0 + m11 - m00 - m22) * 2.0, sq)
    qw = torch.where(cond2, (m02 - m20) / sq, qw)
    qx = torch.where(cond2, (m01 + m10) / sq, qx)
    qy = torch.where(cond2, 0.25 * sq, qy)
    qz = torch.where(cond2, (m12 + m21) / sq, qz)

    # Condition 3
    sq = torch.where(cond3, torch.sqrt(1.0 + m22 - m00 - m11) * 2.0, sq)
    qw = torch.where(cond3, (m10 - m01) / sq, qw)
    qx = torch.where(cond3, (m02 + m20) / sq, qx)
    qy = torch.where(cond3, (m12 + m21) / sq, qy)
    qz = torch.where(cond3, 0.25 * sq, qz)

    quat = torch.stack([qx, qy, qz, qw], dim=-1)

    # Normalize the quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    # Reshape to match input batch shape
    quat = quat.reshape(batch_shape + (4,))

    # If input was a single matrix, remove the batch dimension
    if is_single:
        quat = quat.squeeze(0)

    return quat

# https://behavior.stanford.edu/reference/utils/transform_utils.html#utils.transform_utils.quat2mat
def quat2mat(quaternion):
    """
    Convert quaternions into rotation matrices.

    Args:
        quaternion (torch.Tensor): A tensor of shape (..., 4) representing batches of quaternions (x, y, z, w).

    Returns:
        torch.Tensor: A tensor of shape (..., 3, 3) representing batches of rotation matrices.
    """
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)

    outer = quaternion.unsqueeze(-1) * quaternion.unsqueeze(-2)

    # Extract the necessary components
    xx = outer[..., 0, 0]
    yy = outer[..., 1, 1]
    zz = outer[..., 2, 2]
    xy = outer[..., 0, 1]
    xz = outer[..., 0, 2]
    yz = outer[..., 1, 2]
    xw = outer[..., 0, 3]
    yw = outer[..., 1, 3]
    zw = outer[..., 2, 3]

    rmat = torch.empty(quaternion.shape[:-1] + (3, 3), dtype=quaternion.dtype, device=quaternion.device)

    rmat[..., 0, 0] = 1 - 2 * (yy + zz)
    rmat[..., 0, 1] = 2 * (xy - zw)
    rmat[..., 0, 2] = 2 * (xz + yw)

    rmat[..., 1, 0] = 2 * (xy + zw)
    rmat[..., 1, 1] = 1 - 2 * (xx + zz)
    rmat[..., 1, 2] = 2 * (yz - xw)

    rmat[..., 2, 0] = 2 * (xz - yw)
    rmat[..., 2, 1] = 2 * (yz + xw)
    rmat[..., 2, 2] = 1 - 2 * (xx + yy)

    return rmat