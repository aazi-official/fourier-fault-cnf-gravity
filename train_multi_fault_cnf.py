"""Train and evaluate a response-conditioned CNF for 1 to 4 sequential faults."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
ROOT=Path(__file__).resolve().parent
GEO=ROOT; sys.path.insert(0,str(GEO))
import fourier_fault_prior as geo
from multi_fault_prior import construct_fault_model, validate_model
from train_nf_prior import make_flow
from gravity_forward import compute_sensitivity_gravity

class ResponseCNF(torch.nn.Module):
    def __init__(self, dim, observation_dim=101):
        super().__init__()
        self.embedding=torch.nn.Sequential(torch.nn.Linear(observation_dim,128),torch.nn.SiLU(),torch.nn.Linear(128,64),torch.nn.SiLU())
        self.flow=make_flow(dim, num_layers=8, hidden_features=128, num_bins=8, tail_bound=5., context_features=64)
    def context(self,d): return self.embedding(d)
    def log_prob(self,y,d): return self.flow.log_prob(y, context=self.context(d))
    @torch.no_grad()
    def sample(self,n,d):
        c=self.context(d).expand(n,-1)
        # One independent parameter vector for each observation. nflows returns
        # [batch,num_samples,dim] when a different context is supplied per row.
        return self.flow.sample(1, context=c).squeeze(1)

def reconstruct(theta, times, n_faults):
    p=np.zeros((2,10)); p[:,:6]=theta[:12].reshape(2,6)
    faults=[geo.FaultEvent(*theta[12+3*j:15+3*j], int(times[j])) for j in range(n_faults)]
    return p,faults

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n-faults',type=int,choices=[1,2,3,4,5,6],required=True)
    ap.add_argument('--source',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--models',type=int,default=20000); ap.add_argument('--epochs',type=int,default=150)
    ap.add_argument('--cases',type=int,default=20); ap.add_argument('--samples',type=int,default=4000)
    ap.add_argument('--rho',type=float,nargs=3,default=[.05,-.30,.30],help='Fixed layer density contrasts, never inferred')
    ap.add_argument('--train-only',action='store_true');ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); torch.set_num_threads(4); torch.manual_seed(20260918)
    start=time.time(); src=a.source.resolve(); out=a.out.resolve(); out.mkdir(exist_ok=True)
    data=np.load(src/'valid_models.npz'); split=np.load(src/'split_indices.npz'); theta=data['theta_continuous']; times=data['event_times']; dim=theta.shape[1]
    train,valid,test=[split[k] for k in ['train','validation','test']]
    assert dim==12+3*a.n_faults
    assert np.array_equal(times,np.tile(np.arange(a.n_faults),(len(theta),1))), 'Parameters must already be in event order'
    if a.smoke: train=train[:128]; valid=valid[:32]; test=test[:4]
    nx=nz=24 if a.smoke else 80; xe=np.linspace(0,10000,nx+1); ze=np.linspace(0,5000,nz+1); x=(xe[:-1]+xe[1:])/2; z=(ze[:-1]+ze[1:])/2; xo=np.linspace(0,10000,101); K=compute_sensitivity_gravity(xe,ze,xo,np.full_like(xo,-50)); X,Z=np.meshgrid(x/5000,z/5000); rho=np.asarray(a.rho,dtype=float)
    assert np.all(np.isfinite(rho)), 'Density contrasts must be finite'
    # A response cache is specific to density, source parameters and mesh.
    cache_spec=dict(rho=rho.tolist(),source=str(src),models=len(theta),mesh=[nz,nx],n_faults=a.n_faults)
    cache_path=out/'forward_cache_spec.json'
    if (out/'gravity_clean.npy').exists():
        if cache_path.exists():
            assert json.loads(cache_path.read_text())==cache_spec, 'Forward cache mismatch: choose a fresh output directory'
        else:
            old=json.loads((out/'manifest.json').read_text())
            assert old['rho']==rho.tolist() and old['generator']==str(src) and old['mesh']==[nz,nx] and old['models']==len(theta), 'Legacy cache mismatch'
    cache_path.write_text(json.dumps(cache_spec,indent=2))
    def hard(t,tm):
        p,events=reconstruct(t,tm,a.n_faults); return rho[construct_fault_model(p,events).material(X,Z)]
    response_file=out/'gravity_clean.npy'; done=json.loads((out/'forward_progress.json').read_text())['completed'] if (out/'forward_progress.json').exists() else 0
    responses=np.lib.format.open_memmap(response_file, mode='r+' if response_file.exists() else 'w+', dtype='float64', shape=(len(theta),101))
    for lo in range(done,len(theta),64):
        hi=min(lo+64,len(theta)); models=np.stack([hard(t,tm) for t,tm in zip(theta[lo:hi],times[lo:hi])]); responses[lo:hi]=models.transpose(0,2,1).reshape(len(models),-1)@K.T; responses.flush(); (out/'forward_progress.json').write_text(json.dumps({'completed':hi,'total':len(theta)}))
        if hi%512==0 or hi==len(theta): print(json.dumps({'stage':'forward','completed':hi,'total':len(theta),'elapsed_s':time.time()-start}),flush=True)
    dm=np.asarray(responses[train]).mean(0); ds=np.maximum(np.asarray(responses[train]).std(0),1e-6); ydata=torch.tensor((responses-dm)/ds,dtype=torch.float32); noise=torch.tensor(.1/ds,dtype=torch.float32)
    pmean=torch.tensor(theta[train].mean(0),dtype=torch.float32); pstd=torch.tensor(theta[train].std(0),dtype=torch.float32).clamp_min(1e-4); yparam=(torch.tensor(theta,dtype=torch.float32)-pmean)/pstd
    manifest=dict(n_faults=a.n_faults,models=len(theta),dim=dim,split_sizes=[len(train),len(valid),len(test)],rho=rho.tolist(),sigma_mgal=.1,mesh=[nz,nx],architecture='101->128->64 embedding;8 spline blocks;128 hidden;8 bins;continuous signed parameters;chronology sorted by event age',generator=str(src),no_geological_filter=True,selection='minimum hard-forward relative misfit after sampling',samples_per_case=a.samples,cases=a.cases)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); np.savez(out/'normalization.npz',parameter_mean=pmean.numpy(),parameter_std=pstd.numpy(),data_mean=dm,data_std=ds,train=train,validation=valid,test=test)
    model=ResponseCNF(dim); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5); best=float('inf'); bad=0; hist=[]; maxepoch=3 if a.smoke else a.epochs
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(yparam[train],ydata[train]),batch_size=256,shuffle=True)
    valdata=ydata[valid]+torch.randn_like(ydata[valid])*noise
    for ep in range(1,maxepoch+1):
        model.train(); total=0
        for yp,dd in loader:
            dd=dd+torch.randn_like(dd)*noise; loss=-model.log_prob(yp,dd).mean(); assert torch.isfinite(loss); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); total+=float(loss.detach())*len(yp)
        model.eval();
        with torch.no_grad(): val_loss=float(-torch.cat([model.log_prob(yparam[valid[j:j+256]],valdata[j:j+256]) for j in range(0,len(valid),256)]).mean())
        hist.append([ep,total/len(train),val_loss]); np.save(out/'history.npy',hist); print(json.dumps({'stage':'training','epoch':ep,'train_nll':total/len(train),'validation_nll':val_loss}),flush=True)
        if val_loss<best: best=val_loss;bad=0; torch.save(dict(state=model.state_dict(),mean=pmean,std=pstd,data_mean=dm,data_std=ds,epoch=ep,manifest=manifest),out/'conditional_nf.pt')
        else: bad+=1
        if bad>=20: break
    (out/'training_report.json').write_text(json.dumps(dict(epochs=len(hist),best_validation_nll=best),indent=2))
    if a.smoke or a.train_only: return
    ck=torch.load(out/'conditional_nf.pt',weights_only=False); model.load_state_dict(ck['state']); model.eval(); rows=[]; evout=out/'evaluation_20'; evout.mkdir(exist_ok=True)
    for ci,index in enumerate(test[:a.cases]):
        d=torch.tensor(((responses[index]-dm)/ds)[None],dtype=torch.float32)
        with torch.no_grad(): samples=model.sample(a.samples,d)*(ck['std'])+ck['mean']
        samples=samples.numpy(); obs=np.asarray(responses[index]); truth=hard(theta[index],times[index]); preds=[]; sums=np.zeros_like(truth); sums2=sums.copy(); flags=[]
        for lo in range(0,a.samples,64):
            models=np.stack([hard(t,tm) for t,tm in zip(samples[lo:lo+64],np.tile(np.arange(a.n_faults),(min(64,a.samples-lo),1)))])
            sums+=models.sum(0); sums2+=(models**2).sum(0); preds.append(models.transpose(0,2,1).reshape(len(models),-1)@K.T)
        pred=np.concatenate(preds); mis=np.linalg.norm(pred-obs,axis=1)/np.linalg.norm(obs); bestidx=int(mis.argmin())
        for t in samples:
            p,e=reconstruct(t,np.arange(a.n_faults),a.n_faults); flags.append(validate_model(p,e,nx=401,nz=401)[0])
        avg=sums/a.samples; var=np.maximum(sums2/a.samples-avg**2,0); q=np.quantile(pred,[.05,.5,.95],axis=0); bestmodel=hard(samples[bestidx],np.arange(a.n_faults))
        np.savez_compressed(evout/f'case_{ci+1:02d}.npz',truth_model=truth,bestfit_model=bestmodel,model_mean=avg,model_variance=var,observation=obs,hard_predictions=pred,gravity_quantiles=q,bestfit_index=bestidx,bestfit_misfit=mis[bestidx],valid=flags,truth_theta=theta[index],theta=samples,event_times=np.tile(np.arange(a.n_faults),(a.samples,1)),rho=rho)
        row=dict(case=ci+1,test_index=int(index),bestfit_relative_data_misfit=float(mis[bestidx]),sample_median_misfit=float(np.median(mis)),raw_validity=float(np.mean(flags)),bestfit_valid=bool(flags[bestidx]))
        rows.append(row)
        fig,axs=plt.subplots(1,5,figsize=(22,4),layout='constrained')
        for ax,mat,title in zip(axs[:4],[truth,bestmodel,avg,var],['True','Minimum-misfit model','Pixel mean','Pixel variance']):
            kw=dict(cmap='magma',vmin=0) if title=='Pixel variance' else dict(cmap='viridis',vmin=rho.min(),vmax=rho.max())
            im=ax.imshow(mat,extent=[0,10,5,0],aspect='equal',**kw);ax.set(title=title,xlabel='X (km)',ylabel='Depth (km)');fig.colorbar(im,ax=ax,shrink=.6)
        ax=axs[4];xx=np.linspace(0,10,len(obs));ax.fill_between(xx,q[0],q[2],alpha=.25,label='All samples 5–95%');ax.plot(xx,obs,'k',label='Observed');ax.plot(xx,pred[bestidx],'r--',label='Minimum-misfit forward');ax.set(title=f"Case {ci+1}: misfit {mis[bestidx]*100:.2f}%",xlabel='X (km)',ylabel='Gravity (mGal)');ax.legend(fontsize=7)
        fig.savefig(evout/f'case_{ci+1:02d}.png',dpi=130);plt.close(fig)
        print(json.dumps({'stage':'case_complete',**rows[-1]}),flush=True)
    report=dict(cases=len(rows),mean_bestfit_misfit=float(np.mean([r['bestfit_relative_data_misfit'] for r in rows])),median_bestfit_misfit=float(np.median([r['bestfit_relative_data_misfit'] for r in rows])),mean_validity=float(np.mean([r['raw_validity'] for r in rows])),bestfit_valid_count=sum(r['bestfit_valid'] for r in rows),best_epoch=ck['epoch'],elapsed_s=time.time()-start)
    (evout/'summary.json').write_text(json.dumps(rows,indent=2)); (evout/'report.json').write_text(json.dumps(report,indent=2))
    fig,axs=plt.subplots(len(rows),4,figsize=(16,3*len(rows)),layout='constrained',squeeze=False)
    for i in range(len(rows)):
        d=np.load(evout/f'case_{i+1:02d}.npz');rho=d['rho']
        for ax,key,title in zip(axs[i],['truth_model','bestfit_model','model_mean','model_variance'],['True','Best fit','Mean','Variance']):
            kw=dict(cmap='magma',vmin=0) if key=='model_variance' else dict(cmap='viridis',vmin=rho.min(),vmax=rho.max());ax.imshow(d[key],extent=[0,10,5,0],aspect='auto',**kw);ax.set_title(f"{i+1:02d} {title}")
        axs[i,0].set_ylabel(f"Case {i+1:02d}\nDepth (km)")
    fig.suptitle(f"{a.n_faults}-fault CNF inversion, 20 held-out cases; fixed density; no filtering")
    fig.savefig(evout/'all_20_models.png',dpi=130);plt.close(fig);print(json.dumps(report),flush=True)

if __name__=='__main__': main()
