"""Fresh CNF sampling; no SVGD and no geological rejection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_multi_fault_cnf import ResponseCNF
from verify_case import ROOT, setup, raster

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--samples',type=int,default=4000)
    p.add_argument('--seed',type=int,default=20260918)
    p.add_argument('--out',type=Path,default=ROOT/'runs/inference.npz')
    a=p.parse_args()
    if a.samples < 1: p.error('--samples must be positive')
    if a.out.exists(): p.error('Output exists; choose a new --out path')
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    ck=torch.load(ROOT/'checkpoints/conditional_nf.pt',map_location='cpu',weights_only=False)
    model=ResponseCNF(15);model.load_state_dict(ck['state']);model.eval()
    d=np.load(ROOT/'data/case_20.npz');obs=d['observation']
    context=torch.tensor(((obs-ck['data_mean'])/ck['data_std'])[None],dtype=torch.float32)
    X,Z,G=setup();parameters=[];predictions=[];total=np.zeros((80,80));squared=total.copy()
    best_misfit=float('inf');best_model=None
    for lo in range(0,a.samples,128):
        with torch.no_grad(): theta=(model.sample(min(128,a.samples-lo),context)*ck['std']+ck['mean']).numpy()
        models=np.stack([raster(t,d['rho'],X,Z) for t in theta])
        pred=models.transpose(0,2,1).reshape(len(models),-1)@G.T
        mis=np.linalg.norm(pred-obs,axis=1)/np.linalg.norm(obs);idx=int(mis.argmin())
        if mis[idx]<best_misfit: best_misfit=float(mis[idx]);best_model=models[idx]
        parameters.append(theta);predictions.append(pred);total+=models.sum(0);squared+=(models**2).sum(0)
    mean=total/a.samples;pred=np.concatenate(predictions)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,theta=np.concatenate(parameters),observation=obs,hard_predictions=pred,bestfit_model=best_model,bestfit_misfit=best_misfit,model_mean=mean,model_variance=np.maximum(squared/a.samples-mean**2,0),gravity_quantiles=np.quantile(pred,[.05,.5,.95],axis=0),seed=a.seed,rho=d['rho'])
    print(json.dumps(dict(samples=a.samples,seed=a.seed,best_relative_misfit=best_misfit,output=str(a.out)),indent=2))

if __name__=='__main__': main()
