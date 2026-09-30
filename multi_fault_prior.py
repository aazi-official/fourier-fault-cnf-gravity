"""General sequential geological state and validation for 3/4 2-D faults."""
from itertools import product
import numpy as np
import fourier_fault_prior as geo

MIN_FAULT_SEPARATION = 0.1

def construct_fault_model(boundaries, faults):
    faults = tuple(faults)
    if len(faults) < 1 or len({int(e.event_time) for e in faults}) != len(faults):
        raise ValueError("Fault events must have distinct chronological times")
    state = geo.GeologicalState(np.asarray(boundaries, dtype=float))
    for e in sorted(faults, key=lambda q: q.event_time):
        state = geo.apply_fault_event(state, e)
    return state

def final_interface_branches(state, x):
    """Return consistent final interface branches after all sequential events."""
    x = np.asarray(x, dtype=float)
    out = [[] for _ in range(len(state.boundaries))]
    for mask in product([0, 1], repeat=len(state.events)):
        shift = np.zeros(2)
        for event, side in zip(state.events, mask):
            if side:
                shift += np.asarray(geo.fault_displacement(event.dip, event.throw))
        for j, p in enumerate(state.boundaries):
            z = geo.fourier_boundary(x - shift[0], p) + shift[1]
            sx, sz = state.source_coordinates(x, z)
            consistent = np.isclose(sx, x - shift[0], atol=1e-8, rtol=0)
            consistent &= np.isclose(sz, z - shift[1], atol=1e-8, rtol=0)
            out[j].append(np.where(consistent, z, np.nan))
    return [np.stack(v) for v in out]

def validate_model(boundaries, faults, nx=401, nz=401, origin_range=(.6,1.4), throw_abs_range=(.15,.30), require_vertical_order=True):
    """Validate final geometry; no clipping or parameter repair."""
    faults = tuple(faults)
    errors = []
    info = {"n_faults": len(faults)}
    if len(faults) not in (1, 2, 3, 4, 5, 6):
        return False, {"reasons": ["wrong_fault_count"]}
    if len({int(e.event_time) for e in faults}) != len(faults):
        errors.append("invalid_chronology")
    if not np.isfinite(boundaries).all():
        errors.append("nonfinite_boundary")
    if any(not np.isfinite([e.xf, e.dip, e.throw]).all() for e in faults):
        errors.append("nonfinite_fault")
    if any(not (40 <= abs(e.dip) <= 80 and throw_abs_range[0] <= abs(e.throw) <= throw_abs_range[1] and origin_range[0] <= e.xf <= origin_range[1]) for e in faults):
        errors.append("fault_parameter_range")
    separations = [abs(a.xf - b.xf) for i, a in enumerate(faults) for b in faults[i+1:]]
    info["minimum_origin_separation"] = float(min(separations)) if separations else None
    if any(s < MIN_FAULT_SEPARATION for s in separations):
        errors.append("fault_origins_too_close")
    if errors:
        info["reasons"] = errors
        return False, info

    shifts = [geo.fault_displacement(e.dip, e.throw)[0] for e in faults]
    lo = -sum(max(0, d) for d in shifts)
    hi = 2 - sum(min(0, d) for d in shifts)
    xs = np.linspace(lo, hi, nx)
    gap = geo.fourier_boundary(xs, boundaries[1]) - geo.fourier_boundary(xs, boundaries[0])
    info["prefault_min_gap"] = float(gap.min())
    if gap.min() <= 0:
        errors.append("prefault_crossing")

    state = construct_fault_model(boundaries, faults)
    x = np.linspace(0, 2, nx)
    branches = final_interface_branches(state, x)
    finite = np.concatenate([a[np.isfinite(a)] for a in branches]) if branches else np.array([])
    info["final_depth_min"] = float(finite.min()) if finite.size else None
    info["final_depth_max"] = float(finite.max()) if finite.size else None
    if not finite.size:
        errors.append("missing_final_interfaces")
    elif finite.min() < 0 or finite.max() > 1:
        errors.append("final_interface_out_of_domain")

    if not errors:
        z = np.linspace(0, 1, nz)
        material = state.material(x[None, :], z[:, None]).astype(np.int8)
        if require_vertical_order and np.any(np.diff(material, axis=0) < 0):
            errors.append("final_layer_order_reversal")
        # Preserve the original three-layer requirement without imposing a new
        # pointwise 0.05 thickness floor.
        if any(not np.any(material == label) for label in range(3)):
            errors.append("final_layer_missing")
        else:
            info["top_layer_thickness_proxy"] = float(np.mean((material == 0).sum(0)) / nz)
    info["fault_order"] = np.argsort([e.event_time for e in faults]).tolist()
    info["reasons"] = errors
    return not errors, info
