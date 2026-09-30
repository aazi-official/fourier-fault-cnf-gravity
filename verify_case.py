"""Read-only reconstruction checks for the archived single-fault case."""
from pathlib import Path
import json
import numpy as np
from gravity_forward import compute_sensitivity_gravity
from train_multi_fault_cnf import reconstruct
from multi_fault_prior import construct_fault_model

ROOT = Path(__file__).resolve().parent

def setup():
    edges_x = np.linspace(0, 10000, 81)
    edges_z = np.linspace(0, 5000, 81)
    X, Z = np.meshgrid((edges_x[:-1]+edges_x[1:])/10000,
                       (edges_z[:-1]+edges_z[1:])/10000)
    G = compute_sensitivity_gravity(edges_x, edges_z, np.linspace(0,10000,101), np.full(101,-50.))
    return X, Z, G

def raster(theta, rho, X, Z):
    p, faults = reconstruct(theta, [0], 1)
    return rho[construct_fault_model(p, faults).material(X, Z)]

def main():
    d = np.load(ROOT/'data/case_20.npz')
    X, Z, G = setup()
    truth = raster(d['truth_theta'], d['rho'], X, Z)
    best = raster(d['theta'][int(d['bestfit_index'])], d['rho'], X, Z)
    assert np.array_equal(truth,d['truth_model'])
    assert np.array_equal(best,d['bestfit_model'])
    assert np.allclose(G@truth.ravel(order='F'),d['observation'],atol=1e-9)
    prediction = G@best.ravel(order='F')
    misfit = np.linalg.norm(prediction-d['observation'])/np.linalg.norm(d['observation'])
    assert np.isclose(misfit,float(d['bestfit_misfit']))
    assert int(np.argmin(np.linalg.norm(d['hard_predictions']-d['observation'],axis=1))) == int(d['bestfit_index'])
    print(json.dumps(dict(passed=True,relative_misfit=float(misfit),density_rmse=float(np.sqrt(np.mean((best-truth)**2)))),indent=2))

if __name__ == '__main__':
    main()
