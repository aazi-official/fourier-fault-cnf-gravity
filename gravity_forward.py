import numpy as np
import torch


def compute_sensitivity_gravity(x_edges, z_edges, x_obs, z_obs, gcons=6.672e-3):
    """Python port of the supplied MATLAB ComputeSensitivityGravity.m.

    Parameters
    ----------
    x_edges : (Nx+1,) cell boundaries in metres, increasing
    z_edges : (Nz+1,) cell boundaries in metres, increasing positive downward
    x_obs, z_obs : (Nd,) observation coordinates in metres
    gcons : float
        Kept equal to the user's MATLAB value by default. With density contrasts
        in g/cm^3, K @ density returns mGal (G_SI * 1000 * 1e5).

    Returns
    -------
    K : (Nd, Nz*Nx) float64
        Column ordering exactly matches MATLAB:
        column index = j*Nz + i for cell row i (z) and column j (x).
    """
    x_edges = np.asarray(x_edges, dtype=np.float64)
    z_edges = np.asarray(z_edges, dtype=np.float64)
    x_obs = np.asarray(x_obs, dtype=np.float64)
    z_obs = np.asarray(z_obs, dtype=np.float64)

    nx = len(x_edges) - 1
    nz = len(z_edges) - 1
    nd = len(x_obs)
    K = np.empty((nd, nz * nx), dtype=np.float64)

    for k in range(nd):
        xo = x_obs[k]
        zo = z_obs[k]
        for j in range(nx):
            x1, x2 = x_edges[j], x_edges[j+1]
            dl = xo - x1
            dr = xo - x2
            for i in range(nz):
                ztop, zbtm = z_edges[i], z_edges[i+1]
                dt = ztop - zo
                db = zbtm - zo

                # Equivalent to MATLAB atan(dis/db), but arctan2 is safer numerically.
                # Since db and dt are positive for observations above the mesh,
                # this is equivalent in the intended geometry.
                # At a surface corner, d*log((b²+d²)/(t²+d²)) has
                # limit zero as d -> 0, even when t=0. Avoid 0*inf NaNs.
                f1 = 0.0 if dl == 0 else dl * np.log((db*db + dl*dl) / (dt*dt + dl*dl))
                f2 = 0.0 if dr == 0 else dr * np.log((db*db + dr*dr) / (dt*dt + dr*dr))
                f3 = 2.0 * db * (np.arctan2(dl, db) - np.arctan2(dr, db))
                f4 = 2.0 * dt * (np.arctan2(dl, dt) - np.arctan2(dr, dt))
                K[k, j*nz + i] = gcons * (f1 - f2 + f3 - f4)
    return K


def matlab_vec_np(model_nz_nx):
    """Vectorize (Nz,Nx) exactly as MATLAB model(:)."""
    return np.asarray(model_nz_nx).reshape(-1, order='F')


def matlab_vec_torch(model):
    """Vectorize a torch model using MATLAB column-major cell ordering.

    Input can be [B,1,Nz,Nx], [B,Nz,Nx], or [Nz,Nx].
    Returns [B,Ncell] or [Ncell].
    """
    if model.ndim == 4:
        model = model[:, 0]
        return model.transpose(1, 2).contiguous().reshape(model.shape[0], -1)
    if model.ndim == 3:
        return model.transpose(1, 2).contiguous().reshape(model.shape[0], -1)
    if model.ndim == 2:
        return model.T.contiguous().reshape(-1)
    raise ValueError(f'Unsupported model shape: {tuple(model.shape)}')


def gravity_torch(model, K):
    """Forward gravity for torch model(s), preserving MATLAB vector ordering."""
    mv = matlab_vec_torch(model)
    if mv.ndim == 1:
        return K @ mv
    return mv @ K.T
