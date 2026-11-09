"""Plot the archived single-fault benchmark and its minimum-misfit CNF draw."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    with np.load(ROOT / "data/case_20.npz") as case:
        truth = case["truth_model"]
        inferred = case["bestfit_model"]
        observed = case["observation"]
        predicted = case["hard_predictions"][int(case["bestfit_index"])]
    x = np.linspace(0, 10, observed.size)
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2), sharex=True,
                             sharey="row", layout="constrained")
    for ax, model, title in zip(axes[0], (truth, inferred),
                                ("True model", "CNF inversion")):
        image = ax.imshow(model, extent=(0, 10, 5, 0), aspect="auto",
                          interpolation="nearest", vmin=-0.3, vmax=0.3)
        ax.set_title(title)
    fig.colorbar(image, ax=axes[0].tolist(), label="Density contrast (g/cm³)")
    axes[0, 0].set_ylabel("Depth (km)")
    for ax in axes[1]:
        ax.plot(x, observed, color="black", label="Synthetic observation")
        ax.set_xlabel("Distance (km)")
    axes[1, 1].plot(x, predicted, "--", color="tab:red", label="CNF prediction")
    axes[1, 1].legend(frameon=False, fontsize=9)
    axes[1, 0].set_ylabel("Gravity (mGal)")
    fig.savefig(ROOT / "results/comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
