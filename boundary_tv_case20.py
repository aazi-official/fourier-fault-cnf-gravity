"""Fixed-density, three-layer column-interface inversion; no learned prior."""
from pathlib import Path
import json,time,argparse
import numpy as np
import torch
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gravity_forward import compute_sensitivity_gravity

torch.set_num_threads(2)
ap=argparse.ArgumentParser();ap.add_argument('--smooth',action='store_true');args=ap.parse_args()
P=Path(__file__).resolve().parent;O=P/('boundary_smooth_case20_20260930' if args.smooth else 'boundary_tv_case20_20260930');O.mkdir(exist_ok=False)
lambdas=[.001,.01,.1,1.,10.] if args.smooth else [0.,.001,.01,.1,1.]
t0=time.time();d=np.load(P/'data/case_20.npz')
obs=d['observation'];rho=d['rho'];truth=d['truth_model'];cnf=d['bestfit_model']
G=compute_sensitivity_gravity(np.linspace(0,10000,81),np.linspace(0,5000,81),np.linspace(0,10000,101),np.full(101,-50.))
assert np.allclose(G@truth.ravel(order='F'),obs)
# Prefix integration is exactly equivalent to the original cell operator with
# fractional layer occupation, not a neural or smoothed gravity surrogate.
K=torch.tensor(G.reshape(101,80,80),dtype=torch.float64)
C=torch.cat([torch.zeros((101,80,1),dtype=torch.float64),K.cumsum(2)],2)
col=torch.arange(80)[None,:];station=torch.arange(101)[:,None]
def geometry(u):
    logits=torch.cat([u.reshape(2,80),torch.zeros((1,80),dtype=torch.float64)],0)
    thickness=.02+4.94*torch.softmax(logits,dim=0)
    return thickness.cumsum(0)[:2]
def forward(h):
    def integ(z):
        a=z/.0625;i=torch.floor(a).long().clamp(0,79);f=a-i
        return (C[station,col,i]+f*K[station,col,i]).sum(1)
    return rho[2]*K.sum((1,2))+(rho[0]-rho[1])*integ(h[0])+(rho[1]-rho[2])*integ(h[1])
target=torch.tensor(obs);train=np.arange(101)%5!=2;val=~train
starts=[]
for seed in range(3):
    rng=np.random.default_rng(9000+seed)
    h=np.sort(rng.uniform(.4,4.6,2));th=np.diff([0,*h,5]);q=(th-.02)/4.94
    starts.append(np.repeat(np.log(q[:2]/q[2])[:,None],80,axis=1).ravel())
np.save(O/'initial_coordinates.npy',starts)
records=[];fits=[]
def solve(u,lam,mask,maxiter):
    def objective(v):
        x=torch.tensor(v,requires_grad=True);h=geometry(x);r=(forward(h)-target)/.1
        tv=(torch.sqrt(torch.diff(h,dim=1)**2+.005**2)-.005).sum()
        penalty=(torch.diff(h,dim=1)**2).sum()/.125 if args.smooth else tv
        loss=(r[mask]**2).mean()+lam*penalty
        loss.backward();return float(loss.detach()),x.grad.numpy()
    res=minimize(objective,u,jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-11,'gtol':1e-6,'maxls':40})
    with torch.no_grad():h=geometry(torch.tensor(res.x));g=forward(h).numpy()
    return res,h.numpy(),g
for lam in lambdas:
    for seed,u in enumerate(starts):
        res,h,g=solve(u,lam,train,3000 if args.smooth else 500)
        row=dict(lam=lam,seed=seed,validation_mse=float(np.mean((g[val]-obs[val])**2)),train_mse=float(np.mean((g[train]-obs[train])**2)),iterations=int(res.nit),success=bool(res.success),message=str(res.message))
        records.append(row);fits.append(res.x);print(json.dumps(row),flush=True)
        (O/'progress.json').write_text(json.dumps(records,indent=2))
# Select lambda by mean held-out error over starts, not by truth or CNF fit.
scores={lam:float(np.mean([r['validation_mse'] for r in records if r['lam']==lam])) for lam in lambdas}
chosen=min(scores,key=scores.get);final=[];Hs=[];Gs=[];Ms=[]
for j,row in enumerate(records):
    if not args.smooth and row['lam']!=chosen:continue
    res,h,g=solve(fits[j],row['lam'],np.ones(101,dtype=bool),5000 if args.smooth else 1000)
    z=np.linspace(0,5,81);f1=np.clip((h[0][None,:]-z[:-1,None])/.0625,0,1);f3=np.clip((z[1:,None]-h[1][None,:])/.0625,0,1)
    model=rho[0]*f1+rho[1]*(1-f1-f3)+rho[2]*f3
    assert np.allclose(G@model.ravel(order='F'),g,atol=1e-9)
    assert min(h[0].min(),(h[1]-h[0]).min(),(5-h[1]).min())>=.02-1e-10
    final.append(dict(lam=row['lam'],seed=row['seed'],objective=float(res.fun),roughness=float(np.sum(np.diff(h,axis=1)**2)/.125),relative_misfit=float(np.linalg.norm(g-obs)/np.linalg.norm(obs)),density_rmse=float(np.sqrt(np.mean((model-truth)**2))),success=bool(res.success),iterations=int(res.nit),gradient_max=float(np.max(abs(res.jac))),message=str(res.message)))
    Hs.append(h);Gs.append(g);Ms.append(model)
ib=min((i for i,r in enumerate(final) if r['lam']==chosen),key=lambda i:final[i]['objective']);h=Hs[ib];g=Gs[ib]
report=dict(chosen_lambda=chosen,validation_scores=scores,final=final,selected=ib,elapsed_s=time.time()-t0,cnf_relative_misfit=float(d['bestfit_misfit']),cnf_density_rmse=float(np.sqrt(np.mean((cnf-truth)**2))),selection='interleaved held-out stations; mean validation MSE across 3 starts selects lambda; full-data objective selects start',minimum_layer_km=.02,tv_epsilon_km=.005,uncertainty='none; multistart solutions are not posterior samples')
report.update(penalty='sum squared interface slopes times dx' if args.smooth else 'smoothed TV',lambda_candidates=lambdas)
(O/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(O/'results.npz',interfaces=Hs,predictions=Gs,models=Ms,truth=truth,cnf=cnf,observation=obs,selected=ib)
fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for a,m,title in zip(ax[0],[truth,cnf,Ms[ib]],['Truth','CNF minimum misfit',f'Boundaries + {"smoothness" if args.smooth else "TV"} (lambda={chosen:g})']):
    im=a.imshow(m,extent=[0,10,5,0],aspect='auto',vmin=-.3,vmax=.3);fig.colorbar(im,ax=a);a.set(title=title,xlabel='X (km)',ylabel='Depth (km)')
ax[0,2].plot(np.linspace(.0625,9.9375,80),h.T,'w',lw=.8)
for a,p,title in zip(ax[1,:2],[G@cnf.ravel(order='F'),g],['CNF','Boundary inversion']):
    a.plot(np.linspace(0,10,101),obs,'k',label='Observed');a.plot(np.linspace(0,10,101),p,'r--',label='Predicted');a.legend();a.set(title=f'{title}: {100*np.linalg.norm(p-obs)/np.linalg.norm(obs):.4f}%',xlabel='X (km)',ylabel='Gravity (mGal)')
ax[1,2].plot(range(5),list(scores.values()),'o-');ax[1,2].set_xticks(range(5),[str(k) for k in scores]);ax[1,2].set(yscale='log',xlabel='TV weight',ylabel='Held-out MSE (mGal squared)',title='Truth-independent parameter selection')
fig.savefig(O/'comparison.png',dpi=160);plt.close(fig)
if args.smooth:
    fig,axs=plt.subplots(2,5,figsize=(20,8),layout='constrained')
    for k,lam in enumerate(lambdas):
        j=min((i for i,r in enumerate(final) if r['lam']==lam),key=lambda i:final[i]['objective'])
        axs[0,k].imshow(Ms[j],extent=[0,10,5,0],aspect='auto',vmin=-.3,vmax=.3)
        axs[0,k].set(title=f'lambda={lam:g}; misfit={100*final[j]["relative_misfit"]:.3f}%',xlabel='X (km)',ylabel='Depth (km)')
        axs[1,k].plot(np.linspace(.0625,9.9375,80),Hs[j].T);axs[1,k].set(ylim=(5,0),xlabel='X (km)',ylabel='Interface depth (km)',title=f'Roughness={final[j]["roughness"]:.3f}')
    fig.savefig(O/'smoothing_sweep.png',dpi=150);plt.close(fig)
print(json.dumps(report),flush=True)
