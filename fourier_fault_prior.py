import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from itertools import product

# Two-event API. Existing one-fault entry points remain compatible.
MIN_FAULT_SEPARATION = 0.1

@dataclass(frozen=True)
class FaultEvent:
    xf: float
    dip: float
    throw: float
    event_time: int

    def inverse(self, x, z):
        """Depth positive down; right block moves (throw/tan(dip), throw).

        Signed dip controls left/right inclination. Signed throw controls
        upward/downward motion of the RIGHT block. No sign is discarded.
        """
        right = x >= fault_x_at_z(z, self.xf, self.dip)
        dx, dz = fault_displacement(self.dip, self.throw)
        return x-right*dx, z-right*dz

@dataclass(frozen=True)
class GeologicalState:
    boundaries: np.ndarray
    events: tuple = ()

    def source_coordinates(self, x, z):
        x,z=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(z,dtype=float))
        for event in reversed(self.events):x,z=event.inverse(x,z)
        return x,z

    def material(self, x, z):
        sx,sz=self.source_coordinates(x,z)
        return (sz>=fourier_boundary(sx,self.boundaries[0])).astype(np.uint8)+(sz>=fourier_boundary(sx,self.boundaries[1])).astype(np.uint8)

def apply_fault_event(state, event):
    if state.events and event.event_time<=state.events[-1].event_time:
        raise ValueError('Events must be applied in increasing geological event time')
    return GeologicalState(state.boundaries,state.events+(event,))

def construct_two_fault_model(boundaries, faults):
    if len(faults)!=2 or len({e.event_time for e in faults})!=2:
        raise ValueError('Exactly two faults with distinct event times required')
    state=GeologicalState(np.asarray(boundaries,dtype=float))
    for event in sorted(faults,key=lambda e:e.event_time):state=apply_fault_event(state,event)
    return state

def final_interface_branches(state, horizontal_grid):
    """Enumerate all event-branch preimages; retain only consistent histories.

    No curve is clipped or repaired. Multiple depths per column are retained.
    Branch depths outside the vertical window remain visible to validation.
    """
    x=np.asarray(horizontal_grid);result=[[],[]]
    for mask in product([0,1],repeat=len(state.events)):
        shift=np.sum([np.array(fault_displacement(e.dip,e.throw))*b for e,b in zip(state.events,mask)],axis=0) if mask else np.zeros(2)
        for j,p in enumerate(state.boundaries):
            z=fourier_boundary(x-shift[0],p)+shift[1]
            sx,sz=state.source_coordinates(x,z)
            ok=np.isclose(sx,x-shift[0],atol=1e-9,rtol=0)&np.isclose(sz,z-shift[1],atol=1e-9,rtol=0)
            result[j].append(np.where(ok,z,np.nan))
    return [np.stack(v) for v in result]

def validate_two_fault_model(boundaries, faults, nx=1601, nz=801):
    """Four user constraints; numerical post-event validation, no 0.05 floor."""
    info={};errors=[]
    if len(faults)!=2 or len({e.event_time for e in faults})!=2:return False,{'reasons':['invalid_chronology']}
    if not np.isfinite(boundaries).all() or not all(np.isfinite([e.xf,e.dip,e.throw]).all() for e in faults):return False,{'reasons':['nonfinite']}
    if any(not(40<=abs(e.dip)<=80 and .15<=abs(e.throw)<=.30 and .6<=e.xf<=1.4) for e in faults):errors.append('fault_parameter_range')
    separation=abs(faults[0].xf-faults[1].xf);info['surface_separation']=separation
    if separation<MIN_FAULT_SEPARATION:errors.append('fault_origins_too_close')
    if errors:return False,dict(info,reasons=errors)
    dxs=[fault_displacement(e.dip,e.throw)[0] for e in faults];source_x=np.linspace(-sum(max(0,d) for d in dxs),2-sum(min(0,d) for d in dxs),nx)
    gap=fourier_boundary(source_x,boundaries[1])-fourier_boundary(source_x,boundaries[0]);info['prefault_min_gap']=float(gap.min())
    if gap.min()<=0:return False,dict(info,reasons=['prefault_crossing'])
    state=construct_two_fault_model(boundaries,faults);x=np.linspace(0,2,nx);branches=final_interface_branches(state,x)
    allz=np.concatenate([a[np.isfinite(a)] for a in branches]);info['final_depth_min']=float(allz.min()) if len(allz) else None;info['final_depth_max']=float(allz.max()) if len(allz) else None
    if not len(allz):errors.append('missing_final_interfaces')
    elif allz.min()<0 or allz.max()>1:errors.append('final_interface_out_of_domain')
    # When vertical lines meet fault-contact segments, layer boundaries also
    # include those segments. Validate final material order on the full raster.
    if not errors:
        z=np.linspace(0,1,nz);m=state.material(x[None,:],z[:,None]).astype(np.int8)
        if np.any(np.diff(m,axis=0)<0):errors.append('final_layer_order_reversal')
        if any(not np.all(np.any(m==j,axis=0)) for j in range(3)):errors.append('final_layer_missing_on_grid')
        if not errors:
            b1=z[(m>=1).argmax(0)];b2=z[(m>=2).argmax(0)];info['final_min_middle_thickness_grid']=float((b2-b1).min())
    # Plane traces intersect at the younger event: older line is still straight
    # immediately before that event; afterwards its moved half is offset.
    slopes=[1/np.tan(np.deg2rad(e.dip)) for e in faults];den=slopes[0]-slopes[1]
    zi=(faults[1].xf-faults[0].xf)/den if abs(den)>1e-12 else np.nan;xi=faults[0].xf+zi*slopes[0]
    info['faults_intersect']=bool(np.isfinite(zi) and 0<=zi<=1 and 0<=xi<=2);info['intersection']=[float(xi),float(zi)] if info['faults_intersect'] else None
    info['fault_order']=np.argsort([e.event_time for e in faults]).tolist();info['reasons']=errors
    return not errors,info


# ============================================================
# 1. Global settings
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

# Normalized domain: width : depth = 2 : 1
LX = 2.0
LZ = 1.0

NX = 800
x = np.linspace(0.0, LX, NX)

# Fourier order:
# z0 + k + 4*(an,bn) = 10 parameters per boundary
NF = 4


# ============================================================
# 2. Prior ranges
# ============================================================

# Boundary 1 reference depth
Z01_RANGE = (0.10, 0.40)

# Boundary 2 reference depth
MIN_Z0_SEPARATION = 0.10
Z02_MAX = 0.80

# Overall linear slope
K_RANGE = (-0.10, 0.10)

# Fourier coefficients
FOURIER_SIGMA1 = 0.04
FOURIER_DECAY = 1.5


# ============================================================
# 3. Fault prior
# ============================================================

# Fault starts from z = 0
XF_RANGE = (0.40, 1.60)

# Dip measured from horizontal
DIP_RANGE = (40.0, 80.0)

# Vertical throw
THROW_RANGE = (0.15, 0.30)


# ============================================================
# 4. Geological validity
# ============================================================

MIN_LAYER_THICKNESS = 0.05

Z_MIN_ALLOWED = 0.04
Z_MAX_ALLOWED = 0.96

MAX_ABS_SLOPE = 0.60
MAX_ABS_CURVATURE = 5.0

# At least 8% of boundary must exist on each side of fault
MIN_BLOCK_FRACTION = 0.08


# ============================================================
# 5. Fourier boundary
# ============================================================

def fourier_boundary(x, params, lx=LX):

    z0 = params[0]
    k = params[1]

    z = (
        z0
        + k * (x - lx / 2.0)
    )

    idx = 2

    for n in range(1, NF + 1):

        an = params[idx]
        bn = params[idx + 1]

        omega = (
            2.0
            * np.pi
            * n
            / lx
        )

        z += (
            an * np.cos(omega * x)
            +
            bn * np.sin(omega * x)
        )

        idx += 2

    return z


def fourier_slope(x, params, lx=LX):

    dzdx = np.full_like(
        x,
        params[1],
        dtype=float
    )

    idx = 2

    for n in range(1, NF + 1):

        an = params[idx]
        bn = params[idx + 1]

        omega = (
            2.0
            * np.pi
            * n
            / lx
        )

        dzdx += (
            -an
            * omega
            * np.sin(omega * x)
            +
            bn
            * omega
            * np.cos(omega * x)
        )

        idx += 2

    return dzdx


def fourier_curvature(x, params, lx=LX):

    d2zdx2 = np.zeros_like(
        x,
        dtype=float
    )

    idx = 2

    for n in range(1, NF + 1):

        an = params[idx]
        bn = params[idx + 1]

        omega = (
            2.0
            * np.pi
            * n
            / lx
        )

        d2zdx2 += (
            -an
            * omega**2
            * np.cos(omega * x)
            -
            bn
            * omega**2
            * np.sin(omega * x)
        )

        idx += 2

    return d2zdx2


# ============================================================
# 6. Sample Fourier parameters
# ============================================================

def sample_fourier_coefficients():

    coeffs = []

    for n in range(1, NF + 1):

        sigma_n = (
            FOURIER_SIGMA1
            /
            (n ** FOURIER_DECAY)
        )

        an = rng.normal(
            0.0,
            sigma_n
        )

        bn = rng.normal(
            0.0,
            sigma_n
        )

        coeffs.extend(
            [an, bn]
        )

    return coeffs


def sample_boundary(z0):

    k = rng.uniform(
        *K_RANGE
    )

    coeffs = (
        sample_fourier_coefficients()
    )

    return np.array(
        [z0, k] + coeffs,
        dtype=float
    )


def sample_two_boundaries():

    # Boundary 1
    z01 = rng.uniform(
        *Z01_RANGE
    )

    p1 = sample_boundary(
        z01
    )

    # Boundary 2
    z02_min = (
        z01
        + MIN_Z0_SEPARATION
    )

    if z02_min >= Z02_MAX:
        return None, None

    z02 = rng.uniform(
        z02_min,
        Z02_MAX
    )

    p2 = sample_boundary(
        z02
    )

    return p1, p2


# ============================================================
# 7. Fault generation
# ============================================================

def sample_fault():

    xf = rng.uniform(
        *XF_RANGE
    )

    dip = rng.uniform(
        *DIP_RANGE
    )

    throw = rng.uniform(
        *THROW_RANGE
    )

    return np.array(
        [xf, dip, throw],
        dtype=float
    )


def fault_x_at_z(
    z,
    xf,
    dip_deg
):
    """
    Fault line:

        x_fault(z)
        =
        xf + z/tan(dip)

    Positive z is downward.

    dip = 90° gives a vertical fault.
    """

    dip = np.deg2rad(
        dip_deg
    )

    return (
        xf
        +
        z / np.tan(dip)
    )


def fault_displacement(
    dip_deg,
    throw
):
    """
    throw = vertical displacement.

    Motion is parallel to fault plane.

    dz = throw
    dx = throw/tan(dip)
    """

    dip = np.deg2rad(
        dip_deg
    )

    dz = throw

    dx = (
        throw
        /
        np.tan(dip)
    )

    return dx, dz


# ============================================================
# 8. Split boundary by fault
# ============================================================

def split_boundary_by_fault(
    x,
    z,
    xf,
    dip_deg
):

    # Fault x coordinate at each boundary depth
    x_fault_local = (
        fault_x_at_z(
            z,
            xf,
            dip_deg
        )
    )

    # Right side of fault
    right = (
        x >= x_fault_local
    )

    left = ~right

    return left, right


# ============================================================
# 9. Apply inclined fault displacement
# ============================================================

def apply_fault_to_boundary(
    x,
    z,
    xf,
    dip_deg,
    throw
):
    """
    IMPORTANT:
    We return the left and right branches separately.

    This prevents matplotlib from drawing a fake
    vertical connection across the fault.
    """

    left, right = (
        split_boundary_by_fault(
            x,
            z,
            xf,
            dip_deg
        )
    )

    dx, dz = fault_displacement(
        dip_deg,
        throw
    )

    # Left block unchanged
    x_left = x[left]
    z_left = z[left]

    # Right block displaced along fault plane
    x_right = (
        x[right]
        + dx
    )

    z_right = (
        z[right]
        + dz
    )

    return (
        x_left,
        z_left,
        x_right,
        z_right,
        left,
        right
    )


# ============================================================
# 10. Check original geology
# ============================================================

def check_unfaulted_geometry(
    p1,
    p2,
    verbose=False
):

    z1 = fourier_boundary(
        x,
        p1
    )

    z2 = fourier_boundary(
        x,
        p2
    )


    # --------------------------------------
    # Reference depth
    # --------------------------------------

    if p2[0] <= p1[0]:

        if verbose:
            print(
                "Reject: z02 <= z01"
            )

        return False


    # --------------------------------------
    # Domain
    # --------------------------------------

    if (
        np.min(z1) < Z_MIN_ALLOWED
        or
        np.max(z1) > Z_MAX_ALLOWED
        or
        np.min(z2) < Z_MIN_ALLOWED
        or
        np.max(z2) > Z_MAX_ALLOWED
    ):

        if verbose:
            print(
                "Reject: boundary outside domain"
            )

        return False


    # --------------------------------------
    # Crossing / thickness
    # --------------------------------------

    thickness = (
        z2 - z1
    )

    if np.any(
        thickness <= 0.0
    ):

        if verbose:
            print(
                "Reject: boundary crossing"
            )

        return False


    if (
        np.min(thickness)
        <
        MIN_LAYER_THICKNESS
    ):

        if verbose:
            print(
                "Reject: layer too thin:",
                np.min(thickness)
            )

        return False


    # --------------------------------------
    # Slope
    # --------------------------------------

    slope1 = fourier_slope(
        x,
        p1
    )

    slope2 = fourier_slope(
        x,
        p2
    )

    if (
        np.max(np.abs(slope1))
        >
        MAX_ABS_SLOPE
        or
        np.max(np.abs(slope2))
        >
        MAX_ABS_SLOPE
    ):

        if verbose:
            print(
                "Reject: slope too large"
            )

        return False


    # --------------------------------------
    # Curvature
    # --------------------------------------

    curv1 = fourier_curvature(
        x,
        p1
    )

    curv2 = fourier_curvature(
        x,
        p2
    )

    if (
        np.max(np.abs(curv1))
        >
        MAX_ABS_CURVATURE
        or
        np.max(np.abs(curv2))
        >
        MAX_ABS_CURVATURE
    ):

        if verbose:
            print(
                "Reject: curvature too large"
            )

        return False


    return True


# ============================================================
# 11. Check fault
# ============================================================

def check_faulted_geometry(
    p1,
    p2,
    fault,
    verbose=False
):

    xf, dip, throw = fault

    z1 = fourier_boundary(
        x,
        p1
    )

    z2 = fourier_boundary(
        x,
        p2
    )

    (
        x1L,
        z1L,
        x1R,
        z1R,
        left1,
        right1
    ) = apply_fault_to_boundary(
        x,
        z1,
        xf,
        dip,
        throw
    )

    (
        x2L,
        z2L,
        x2R,
        z2R,
        left2,
        right2
    ) = apply_fault_to_boundary(
        x,
        z2,
        xf,
        dip,
        throw
    )


    # ========================================================
    # 1. Make sure both fault blocks exist
    # ========================================================

    branches = [
        z1L,
        z1R,
        z2L,
        z2R
    ]

    for branch in branches:

        if branch.size == 0:

            if verbose:
                print(
                    "Reject: empty fault block"
                )

            return False


    # ========================================================
    # 2. Vertical domain after fault
    #
    # x outside [0,2] is NOT rejected.
    # It will simply be clipped by the model window.
    # ========================================================

    for branch in branches:

        if np.min(branch) < Z_MIN_ALLOWED:

            if verbose:
                print(
                    "Reject: faulted boundary above z domain"
                )

            return False


        if np.max(branch) > Z_MAX_ALLOWED:

            if verbose:
                print(
                    "Reject: faulted boundary below z domain"
                )

            return False


    # ========================================================
    # 3. Fault cannot be too close to edge
    # ========================================================

    n_left = np.sum(
        left1
    )

    n_right = np.sum(
        right1
    )

    min_points = int(
        MIN_BLOCK_FRACTION
        * NX
    )

    if (
        n_left < min_points
        or
        n_right < min_points
    ):

        if verbose:
            print(
                "Reject: fault too close to model edge"
            )

        return False


    return True


# ============================================================
# 12. Check faulted geometry
# ============================================================

def check_faulted_geometry(
    p1,
    p2,
    fault,
    verbose=False
):

    xf, dip, throw = fault


    z1 = fourier_boundary(
        x,
        p1
    )

    z2 = fourier_boundary(
        x,
        p2
    )


    (
        x1L,
        z1L,
        x1R,
        z1R,
        left1,
        right1
    ) = apply_fault_to_boundary(
        x,
        z1,
        xf,
        dip,
        throw
    )


    (
        x2L,
        z2L,
        x2R,
        z2R,
        left2,
        right2
    ) = apply_fault_to_boundary(
        x,
        z2,
        xf,
        dip,
        throw
    )


    # ========================================================
    # Vertical bounds
    # ========================================================

    branches_z = [
        z1L,
        z1R,
        z2L,
        z2R
    ]


    for zb in branches_z:

        if len(zb) == 0:

            if verbose:
                print(
                    "Reject: empty fault block"
                )

            return False


        if (
            np.min(zb)
            <
            Z_MIN_ALLOWED
            or
            np.max(zb)
            >
            Z_MAX_ALLOWED
        ):

            if verbose:
                print(
                    "Reject: faulted geometry outside z domain"
                )

            return False


    # ========================================================
    # Ensure both sides are visible
    # ========================================================

    n_left = np.sum(
        left1
    )

    n_right = np.sum(
        right1
    )


    min_points = int(
        MIN_BLOCK_FRACTION
        * NX
    )


    if (
        n_left
        <
        min_points
        or
        n_right
        <
        min_points
    ):

        if verbose:
            print(
                "Reject: fault too close to edge"
            )

        return False


    return True


# ============================================================
# 13. Complete validity
# ============================================================

def check_fault_geometry(fault, verbose=False):
    """Check that the inclined fault intersects the model window."""
    xf, dip, throw = fault

    if not (0.0 < dip < 90.0) or throw <= 0.0:
        if verbose:
            print("Reject: invalid fault parameters")
        return False

    ztest = np.linspace(0.0, LZ, 500)
    xfault = fault_x_at_z(ztest, xf, dip)
    if not np.any((xfault >= 0.0) & (xfault <= LX)):
        if verbose:
            print("Reject: fault misses model domain")
        return False

    return True

def is_valid_model(
    p1,
    p2,
    fault,
    verbose=False
):

    if not check_unfaulted_geometry(
        p1,
        p2,
        verbose
    ):

        return False


    if not check_fault_geometry(
        fault,
        verbose
    ):

        return False


    if not check_faulted_geometry(
        p1,
        p2,
        fault,
        verbose
    ):

        return False


    return True


# ============================================================
# 14. Generate one model
# ============================================================

def generate_one_valid_model(
    max_trials=100000
):

    for _ in range(
        max_trials
    ):

        p1, p2 = (
            sample_two_boundaries()
        )


        if p1 is None:
            continue


        fault = (
            sample_fault()
        )


        if is_valid_model(
            p1,
            p2,
            fault
        ):

            return np.concatenate(
                [
                    p1,
                    p2,
                    fault
                ]
            )


    raise RuntimeError(
        "Cannot generate valid model."
    )


# ============================================================
# 15. Decode
# ============================================================

def decode_theta(theta):

    theta = np.asarray(
        theta
    )


    if theta.size != 23:

        raise ValueError(
            "theta must contain 23 parameters."
        )


    p1 = theta[:10]
    p2 = theta[10:20]
    fault = theta[20:23]


    return (
        p1,
        p2,
        fault
    )


# ============================================================
# 16. Plot
# ============================================================

def plot_model(
    theta,
    save_path=None,
    show=True
):

    p1, p2, fault = (
        decode_theta(theta)
    )


    xf, dip, throw = fault


    z1 = fourier_boundary(
        x,
        p1
    )

    z2 = fourier_boundary(
        x,
        p2
    )


    (
        x1L,
        z1L,
        x1R,
        z1R,
        _,
        _
    ) = apply_fault_to_boundary(
        x,
        z1,
        xf,
        dip,
        throw
    )


    (
        x2L,
        z2L,
        x2R,
        z2R,
        _,
        _
    ) = apply_fault_to_boundary(
        x,
        z2,
        xf,
        dip,
        throw
    )


    # ========================================================
    # Fault line
    # ========================================================

    z_fault = np.linspace(
        0.0,
        LZ,
        1000
    )


    x_fault = (
        fault_x_at_z(
            z_fault,
            xf,
            dip
        )
    )


    mask = (
        (x_fault >= 0.0)
        &
        (x_fault <= LX)
    )


    # ========================================================
    # Plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )


    # Boundary 1, two separate branches
    ax.plot(
        x1L,
        z1L,
        linewidth=2.5,
        label="Boundary 1"
    )

    ax.plot(
        x1R,
        z1R,
        linewidth=2.5
    )


    # Boundary 2
    ax.plot(
        x2L,
        z2L,
        linewidth=2.5,
        label="Boundary 2"
    )

    ax.plot(
        x2R,
        z2R,
        linewidth=2.5
    )


    # Fault
    ax.plot(
        x_fault[mask],
        z_fault[mask],
        "--",
        linewidth=2.5,
        label="Fault"
    )


    ax.set_xlim(
        0,
        LX
    )


    ax.set_ylim(
        LZ,
        0
    )


    ax.set_aspect(
        "equal",
        adjustable="box"
    )


    ax.set_xlabel(
        "Normalized X"
    )

    ax.set_ylabel(
        "Normalized Depth"
    )


    dx, dz = (
        fault_displacement(
            dip,
            throw
        )
    )


    ax.set_title(
        "Fourier Layer Model with Inclined Fault\n"
        f"xf={xf:.3f}, "
        f"dip={dip:.1f}°, "
        f"throw={throw:.3f}, "
        f"dx={dx:.3f}"
    )


    ax.legend()

    ax.grid(
        alpha=0.15
    )


    plt.tight_layout()


    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


    if show:
        plt.show()


    plt.close(fig)


# ============================================================
# 17. Generate dataset
# ============================================================

def generate_dataset(
    n_valid=100000,
    max_total_trials=5000000
):

    theta_list = []

    n_trials = 0


    while (
        len(theta_list)
        <
        n_valid
    ):

        n_trials += 1


        if (
            n_trials
            >
            max_total_trials
        ):

            raise RuntimeError(
                "Maximum trials exceeded."
            )


        p1, p2 = (
            sample_two_boundaries()
        )


        if p1 is None:
            continue


        fault = (
            sample_fault()
        )


        if not is_valid_model(
            p1,
            p2,
            fault
        ):

            continue


        theta_list.append(
            np.concatenate(
                [
                    p1,
                    p2,
                    fault
                ]
            )
        )


        if (
            len(theta_list)
            %
            5000
            == 0
        ):

            print(
                f"Valid: "
                f"{len(theta_list):8d}"
                f" | Trials: "
                f"{n_trials:9d}"
                f" | Acceptance: "
                f"{len(theta_list)/n_trials:.4f}"
            )


    theta_array = np.asarray(
        theta_list,
        dtype=np.float32
    )


    print()
    print(
        "=============================="
    )

    print(
        "Dataset generation completed"
    )

    print(
        "=============================="
    )

    print(
        "theta shape:",
        theta_array.shape
    )

    print(
        "Total trials:",
        n_trials
    )

    print(
        "Acceptance rate:",
        len(theta_array)
        /
        n_trials
    )


    return theta_array


# ============================================================
# 18. Main
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # First generate ONE model so you can verify the fault
    # ========================================================

    theta_test = (
        generate_one_valid_model()
    )


    print()
    print(
        "Test model:"
    )

    print(
        theta_test
    )


    print()
    print(
        "Fault parameters:"
    )

    print(
        theta_test[20:23]
    )


    plot_model(
        theta_test,
        save_path="inclined_fault_test.png",
        show=True
    )


    # ========================================================
    # After confirming the figure, generate 100000
    # ========================================================

    theta_train = (
        generate_dataset(
            n_valid=100000
        )
    )


    # ========================================================
    # Save
    # ========================================================

    np.savez_compressed(
        "fourier_fault_valid_prior.npz",

        theta=theta_train,

        lx=LX,
        lz=LZ,

        nf=NF,

        x=x,

        z01_range=
            np.array(Z01_RANGE),

        min_z0_separation=
            MIN_Z0_SEPARATION,

        z02_max=
            Z02_MAX,

        k_range=
            np.array(K_RANGE),

        fourier_sigma1=
            FOURIER_SIGMA1,

        fourier_decay=
            FOURIER_DECAY,

        xf_range=
            np.array(XF_RANGE),

        dip_range=
            np.array(DIP_RANGE),

        throw_range=
            np.array(THROW_RANGE),

        min_layer_thickness=
            MIN_LAYER_THICKNESS,

        z_min_allowed=
            Z_MIN_ALLOWED,

        z_max_allowed=
            Z_MAX_ALLOWED,

        max_abs_slope=
            MAX_ABS_SLOPE,

        max_abs_curvature=
            MAX_ABS_CURVATURE,

        fault_displacement_mode=
            "along_fault_plane_vertical_throw"
    )


    print()
    print(
        "Saved:"
    )

    print(
        "fourier_fault_valid_prior.npz"
    )
