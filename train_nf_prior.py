#!/usr/bin/env python3
"""
Train a differentiable Normalizing-Flow geological prior for the
23-D Fourier-boundary + fault parameterization.

Input:
    fourier_fault_valid_prior.npz
        theta[:, :10]   boundary 1
        theta[:, 10:20] boundary 2
        theta[:, 20]    fault x at z=0
        theta[:, 21]    fault dip (degrees)
        theta[:, 22]    fault throw

Outputs:
    nf_prior_checkpoint.pt
    nf_training_history.npz
    nf_samples.npz
    nf_diagnostics.png

Dependencies:
    pip install numpy matplotlib torch nflows
"""

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from nflows.distributions.normal import StandardNormal
    from nflows.flows.base import Flow
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.permutations import RandomPermutation
    from nflows.transforms.autoregressive import (
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing package 'nflows'. Install with:\n"
        "  pip install nflows\n"
        "or:\n"
        "  python3 -m pip install nflows"
    ) from exc


PARAM_NAMES = [
    "z01","k1","a11","b11","a12","b12","a13","b13","a14","b14",
    "z02","k2","a21","b21","a22","b22","a23","b23","a24","b24",
    "xf","dip","throw"
]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_flow(dim, num_layers=6, hidden_features=128, num_bins=8, tail_bound=4.0, context_features=None):
    transforms = []
    for _ in range(num_layers):
        transforms.append(RandomPermutation(features=dim))
        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_features,
                context_features=context_features,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                num_blocks=2,
                use_residual_blocks=True,
                random_mask=False,
                activation=torch.relu,
                dropout_probability=0.0,
                use_batch_norm=False,
            )
        )
    return Flow(
        CompositeTransform(transforms),
        StandardNormal([dim]),
    )


def fourier_boundary(x, p, lx, nf):
    z = p[0] + p[1] * (x - lx / 2.0)
    j = 2
    for n in range(1, nf + 1):
        w = 2.0 * np.pi * n / lx
        z = z + p[j] * np.cos(w*x) + p[j+1] * np.sin(w*x)
        j += 2
    return z


def fourier_slope(x, p, lx, nf):
    dz = np.full_like(x, p[1], dtype=float)
    j = 2
    for n in range(1, nf + 1):
        w = 2.0 * np.pi * n / lx
        dz += -p[j]*w*np.sin(w*x) + p[j+1]*w*np.cos(w*x)
        j += 2
    return dz


def fourier_curvature(x, p, lx, nf):
    d2 = np.zeros_like(x, dtype=float)
    j = 2
    for n in range(1, nf + 1):
        w = 2.0 * np.pi * n / lx
        d2 += -p[j]*(w**2)*np.cos(w*x) - p[j+1]*(w**2)*np.sin(w*x)
        j += 2
    return d2


def fault_x_at_z(z, xf, dip_deg):
    return xf + z / np.tan(np.deg2rad(dip_deg))


def apply_fault(x, z, xf, dip, throw):
    right = x >= fault_x_at_z(z, xf, dip)
    dx = throw / np.tan(np.deg2rad(dip))
    xf_coords = x.copy()
    zf = z.copy()
    xf_coords[right] += dx
    zf[right] += throw
    return xf_coords, zf, right


def is_valid(theta, lx, lz, nf,
             min_thickness=0.12,
             zmin=0.04, zmax=0.96,
             max_slope=0.60,
             max_curvature=5.0):
    """
    Same basic validity logic used by the generator.
    Returns False rather than clipping an NF sample.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape[-1] != 23 or not np.all(np.isfinite(theta)):
        return False

    p1 = theta[:10]
    p2 = theta[10:20]
    xf, dip, throw = theta[20:23]

    if p2[0] <= p1[0]:
        return False
    if not (0.0 < dip < 90.0) or throw <= 0.0:
        return False

    x = np.linspace(0.0, lx, 800)
    z1 = fourier_boundary(x, p1, lx, nf)
    z2 = fourier_boundary(x, p2, lx, nf)

    if min(z1.min(), z2.min()) < zmin:
        return False
    if max(z1.max(), z2.max()) > zmax:
        return False
    if np.min(z2-z1) < min_thickness:
        return False

    if np.max(np.abs(fourier_slope(x,p1,lx,nf))) > max_slope:
        return False
    if np.max(np.abs(fourier_slope(x,p2,lx,nf))) > max_slope:
        return False
    if np.max(np.abs(fourier_curvature(x,p1,lx,nf))) > max_curvature:
        return False
    if np.max(np.abs(fourier_curvature(x,p2,lx,nf))) > max_curvature:
        return False

    ztest = np.linspace(0.0, lz, 500)
    xfault = fault_x_at_z(ztest, xf, dip)
    if not np.any((xfault >= 0.0) & (xfault <= lx)):
        return False

    x1f, z1f, side1 = apply_fault(x, z1, xf, dip, throw)
    x2f, z2f, _ = apply_fault(x, z2, xf, dip, throw)

    if min(z1f.min(), z2f.min()) < zmin:
        return False
    if max(z1f.max(), z2f.max()) > zmax:
        return False
    if np.min(z2f-z1f) < min_thickness:
        return False

    min_points = int(0.08 * len(x))
    nr = int(side1.sum())
    if nr < min_points or (len(x)-nr) < min_points:
        return False

    return True


def sample_valid(prior, n_samples, lx, lz, nf, batch_size=2048,
                 max_draws=None):
    """Draw from the NF with rejection sampling and return valid models only."""
    if max_draws is None:
        max_draws = max(100 * n_samples, 1_000_000)

    accepted = []
    draws = 0
    while sum(len(a) for a in accepted) < n_samples and draws < max_draws:
        n = min(batch_size, max_draws - draws)
        batch = prior.sample(n).cpu().numpy()
        draws += n
        flags = np.asarray([
            is_valid(th, lx, lz, nf) for th in batch
        ], dtype=bool)
        if flags.any():
            accepted.append(batch[flags])

    if not accepted:
        raise RuntimeError("NF rejection sampling produced no valid models.")

    samples = np.concatenate(accepted, axis=0)[:n_samples]
    return samples, draws, len(samples) / draws


class GeologicalNFPrior:
    """
    Convenience wrapper for use later inside SVGD.

    Important:
    log_prob(theta) is the density in ORIGINAL theta coordinates.
    score(theta) returns d log p(theta) / d theta.
    """
    def __init__(self, flow, mean, std, device):
        self.flow = flow.to(device)
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.as_tensor(std, dtype=torch.float32, device=device)
        self.device = device

    def log_prob(self, theta):
        if not torch.is_tensor(theta):
            theta = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
        else:
            theta = theta.to(self.device)

        ztheta = (theta - self.mean) / self.std
        # Change of variables for standardization:
        # p_theta(theta) = p_y(y) * prod_j 1/std_j
        return self.flow.log_prob(ztheta) - torch.log(self.std).sum()

    def score(self, theta):
        if not torch.is_tensor(theta):
            theta = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
        theta = theta.to(self.device).detach().clone().requires_grad_(True)
        lp = self.log_prob(theta).sum()
        grad = torch.autograd.grad(lp, theta)[0]
        return grad

    @torch.no_grad()
    def sample(self, n):
        y = self.flow.sample(n)
        return y * self.std + self.mean


def save_checkpoint(path, flow, mean, std, config):
    torch.save({
        "state_dict": flow.state_dict(),
        "mean": np.asarray(mean, dtype=np.float32),
        "std": np.asarray(std, dtype=np.float32),
        "config": config,
        "param_names": PARAM_NAMES,
    }, path)


def load_prior(checkpoint, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(
    checkpoint,
    map_location=device,
    weights_only=False
)
    cfg = ckpt["config"]
    flow = make_flow(
        cfg["dim"], cfg["num_layers"], cfg["hidden_features"],
        cfg["num_bins"], cfg["tail_bound"]
    )
    flow.load_state_dict(ckpt["state_dict"])
    flow.eval()
    return GeologicalNFPrior(flow, ckpt["mean"], ckpt["std"], device)


def main():
    print('LEGACY TRAINER: its validity checker does not match the along-plane '
          'generator. Use train_prior_svgd.py for the new dataset and SVGD prior.', flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="fourier_fault_valid_prior.npz")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--tail-bound", type=float, default=4.0)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--nf-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="nf_prior_output")
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data)
    theta = np.asarray(data["theta"], dtype=np.float32)
    lx = float(data["lx"])
    lz = float(data["lz"])
    nf = int(data["nf"])

    if theta.ndim != 2 or theta.shape[1] != 23:
        raise ValueError(f"Expected theta shape (N,23), got {theta.shape}")

    print(f"Loaded theta: {theta.shape}")
    print(f"Domain LX={lx}, LZ={lz}, Fourier order={nf}")

    # Reproducible 80/20 split.
    idx = np.random.permutation(len(theta))
    ntrain = int(0.8 * len(theta))
    train = theta[idx[:ntrain]]
    valid = theta[idx[ntrain:]]

    # IMPORTANT: fit normalization on TRAINING data only.
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=0)
    std = np.maximum(std, 1e-8)

    train_z = (train - mean) / std
    valid_z = (valid - mean) / std

    print(f"Train: {train.shape}; validation: {valid.shape}")
    print("Normalization fitted on training split only.")

    # Save basic diagnostics before training.
    corr = np.corrcoef(train.T)
    np.save(outdir / "training_correlation.npy", corr)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_title("Accepted-prior parameter correlation")
    ax.set_xlabel("Parameter index")
    ax.set_ylabel("Parameter index")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "correlation_matrix.png", dpi=180)
    plt.close(fig)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_t = torch.from_numpy(train_z.astype(np.float32))
    valid_t = torch.from_numpy(valid_z.astype(np.float32)).to(device)

    loader = DataLoader(
        TensorDataset(train_t),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    config = {
        "dim": 23,
        "num_layers": args.layers,
        "hidden_features": args.hidden,
        "num_bins": args.bins,
        "tail_bound": args.tail_bound,
        "lx": lx,
        "lz": lz,
        "nf": nf,
    }

    flow = make_flow(
        23, args.layers, args.hidden, args.bins, args.tail_bound
    ).to(device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr)

    train_hist = []
    valid_hist = []
    best_val = math.inf
    best_epoch = -1
    wait = 0
    best_path = outdir / "nf_prior_checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        flow.train()
        total = 0.0
        count = 0

        for (xb,) in loader:
            xb = xb.to(device)
            loss = -flow.log_prob(xb).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=10.0)
            optimizer.step()

            total += float(loss.item()) * len(xb)
            count += len(xb)

        train_loss = total / count

        flow.eval()
        with torch.no_grad():
            val_loss = float((-flow.log_prob(valid_t).mean()).item())

        train_hist.append(train_loss)
        valid_hist.append(val_loss)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            save_checkpoint(best_path, flow, mean, std, config)
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d} | train NLL={train_loss:9.5f} | "
                f"val NLL={val_loss:9.5f} | best={best_val:9.5f}"
            )

        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    np.savez(
        outdir / "nf_training_history.npz",
        train_nll=np.asarray(train_hist),
        valid_nll=np.asarray(valid_hist),
        best_epoch=best_epoch,
        best_valid_nll=best_val,
    )

    # Reload best model.
    prior = load_prior(best_path, device=device)

    # Test original-coordinate density and score.
    test_theta = torch.as_tensor(valid[:8], dtype=torch.float32, device=device)
    logp = prior.log_prob(test_theta)
    score = prior.score(test_theta)
    print("\nNF API test:")
    print("log_prob shape:", tuple(logp.shape))
    print("score shape:   ", tuple(score.shape))
    print("finite logp:   ", bool(torch.isfinite(logp).all()))
    print("finite score:  ", bool(torch.isfinite(score).all()))

    # Draw raw NF samples first to measure the unconstrained acceptance rate.
    raw_samples = prior.sample(args.nf_samples).cpu().numpy()
    raw_good = np.asarray([
        is_valid(th, lx, lz, nf) for th in raw_samples
    ], dtype=bool)
    raw_validity_rate = float(raw_good.mean())
    print(
        f"\nRaw NF geological validity: "
        f"{raw_good.sum()}/{len(raw_good)} = {raw_validity_rate:.4f}"
    )

    # Rejection-sampled output for downstream SVGD initialization.
    samples, draws, rejection_rate = sample_valid(
        prior, args.nf_samples, lx, lz, nf
    )
    good = np.ones(len(samples), dtype=bool)
    print(
        f"NF valid sample set: {len(samples)}/{draws} accepted "
        f"(rejection-sampling rate={rejection_rate:.4f})"
    )

    np.savez(
        outdir / "nf_samples.npz",
        theta=samples,
        is_valid=good,
        validity_rate=float(good.mean()),
        raw_validity_rate=raw_validity_rate,
        raw_samples=raw_samples,
        raw_is_valid=raw_good,
        rejection_draws=draws,
    )

    # Training curves.
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, len(train_hist)+1), train_hist, label="train")
    ax.plot(np.arange(1, len(valid_hist)+1), valid_hist, label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative log likelihood")
    ax.set_title("Normalizing-flow training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "training_curve.png", dpi=180)
    plt.close(fig)

    # Compare a few key marginals: accepted training data vs NF.
    cols = [0, 10, 20, 21, 22]
    labels = ["z01", "z02", "fault x", "fault dip", "fault throw"]
    fig, axes = plt.subplots(len(cols), 1, figsize=(8, 13))
    for ax, col, lab in zip(axes, cols, labels):
        ax.hist(train[:, col], bins=40, density=True, alpha=0.55, label="accepted data")
        ax.hist(samples[:, col], bins=40, density=True, alpha=0.55, label="NF samples")
        ax.set_xlabel(lab)
        ax.set_ylabel("density")
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "marginal_comparison.png", dpi=180)
    plt.close(fig)

    print("\nSaved:")
    print(" ", best_path)
    print(" ", outdir / "nf_training_history.npz")
    print(" ", outdir / "nf_samples.npz")
    print(" ", outdir / "correlation_matrix.png")
    print(" ", outdir / "training_curve.png")
    print(" ", outdir / "marginal_comparison.png")

    print("\nLater SVGD usage:")
    print("  prior = load_prior('nf_prior_output/nf_prior_checkpoint.pt')")
    print("  logp = prior.log_prob(theta)")
    print("  grad_prior = prior.score(theta)")


if __name__ == "__main__":
    main()
