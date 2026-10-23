"""Opt-in real CUDA sampling + isolated CPU/GPU fit and paired-bin acceptance run.

Run: OMP_NUM_THREADS=1 python tests/cuda_smoke.py /tmp/cpn-cuda-report.json
Uses small explicit smoke budgets; never changes production config or bins.
"""
from pathlib import Path
import json
import sys
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from cpn_renorm.config import load_config, mod_config
from cpn_renorm.patches import generate_one_plaq, generate_two_plaq
from cpn_renorm.fit_client import isolated_fit
from cpn_renorm.exact_fitting import Conditional
from cpn_renorm.torch_fitting import Conditional as GPUConditional, OnePlaquette
from cpn_renorm.fitting import one_plaq_probability


def main(destination):
    started=time.perf_counter()
    torch.set_num_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA smoke requires visible NVIDIA device')
    cfg=mod_config(load_config(Path(__file__).parents[1]/'configs/example.toml'),1)
    cfg['runtime'].update(device='cuda',dtype='float64',auto_chains=False)
    cfg['model'].update(N=2,beta=.7,beta1=.0,alpha=.3,alpha1=-.1)
    cfg['renormalization'].update(factor=2,type=['U','z'])
    cfg['geometry'].update(boundary_bc='OBC',padding=1)
    for stage in ('one_plaq','two_plaq'):
        cfg['chains'][stage]=4
        cfg['steps'][stage].update(fit_times=4,min_warmup=16,max_warmup=32,
            min_meas_per_boundary=64,max_meas_per_boundary=128,target_ess_per_boundary=10.)
    cfg['hmc'].update(adapt_steps=8)
    geometry={'padding':1,'factor':2,'L_fine':8,'L_coarse':4}
    report={'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'sampling':{},'fits':{}}
    with tempfile.TemporaryDirectory() as tmp:
        for stage,generate in (('one_plaq',generate_one_plaq),('two_plaq',generate_two_plaq)):
            t=time.perf_counter()
            raw=generate(cfg,geometry,Path(tmp)/stage)
            report['sampling'][stage]={'seconds':time.perf_counter()-t,
                **{key:np.asarray(raw[key]).tolist() for key in ('accept_rate','sampling_ess','sampling_tau',
                    'sampling_actual','sampling_stop_reason','warmup_stop_reason')}}
            for tag in ('U','z'):
                key='s' if stage=='one_plaq' and tag=='U' else tag
                X,freq=raw['X_'+key],raw['freq_'+key]
                request={'p0':.4} if stage=='one_plaq' else {'N':2,'alpha':report['fits']['one_plaq_'+tag]['cpu']['alpha'],'p0':[.6,.1]}
                operation='one_plaq' if stage=='one_plaq' else 'two_fixed'
                pair={device:isolated_fit(operation,{**request,'device':device},X=X,freq=freq) for device in ('cpu','cuda')}
                cpu,gpu=pair['cpu'],pair['cuda']
                if stage=='one_plaq':
                    np.testing.assert_allclose([cpu['alpha'],cpu['alpha_err'],cpu['mse']],
                                               [gpu['alpha'],gpu['alpha_err'],gpu['mse']],rtol=2e-4,atol=2e-7)
                else:
                    for variant in ('old','new'):
                        names=['beta','mse','beta_err']+(['alpha1','alpha1_err'] if variant=='new' else ['alpha_eff','alpha_eff_err'])
                        np.testing.assert_allclose([cpu[variant][name] for name in names],
                                                   [gpu[variant][name] for name in names],rtol=2e-3,atol=1e-6)
                    pred=[Conditional(2,X).predict([r['new']['beta'],request['alpha'],r['new']['alpha1']]) for r in (cpu,gpu)]
                    pair['max_prediction_difference']=float(np.max(abs(pred[0]-pred[1])))
                if stage == 'one_plaq':
                    model = OnePlaquette(X, 'cuda')
                    pred, jac = model.evaluate(model.tensor([cpu['alpha']]))
                    reference, derivative = one_plaq_probability(X, cpu['alpha'])
                    np.testing.assert_allclose(pred.cpu().numpy(), reference, rtol=1e-10, atol=2e-11)
                    np.testing.assert_allclose(jac[:,0].cpu().numpy(), derivative, rtol=1e-9, atol=2e-10)
                    same_prediction_delta = float(np.max(abs(pred.cpu().numpy()-reference)))
                else:
                    parameters = [cpu['new']['beta'], request['alpha'], cpu['new']['alpha1']]
                    model = GPUConditional(2, X, device='cuda')
                    reference = Conditional(2, X).predict(parameters)
                    actual = model.predict(parameters).cpu().numpy()
                    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=2e-11)
                    same_prediction_delta = float(np.max(abs(actual-reference)))
                    diagnostic = {device: isolated_fit('two_mse', {'N':2, 'parameters':parameters,
                        'device':device}, X=X, freq=freq)['mse'] for device in ('cpu','cuda')}
                    np.testing.assert_allclose(diagnostic['cpu'], diagnostic['cuda'], rtol=1e-10, atol=2e-12)
                    del model
                pair['cpu']['same_parameter_density_max_difference'] = same_prediction_delta
                report['fits'][stage+'_'+tag]={'cpu':cpu,'cuda':gpu,**({} if stage=='one_plaq' else {'max_prediction_difference':pair['max_prediction_difference']})}
    # Same generated angle samples, grouped into B and 2B histograms.
    rng=np.random.default_rng(20261022)
    boundaries=rng.uniform(-np.pi,np.pi,(6,8));boundaries[:,6]=.6
    theta=-np.pi+(np.arange(8192)+.5)*2*np.pi/8192
    x=np.column_stack((np.tile(theta,6),np.repeat(boundaries,8192,axis=0)))
    density=Conditional(2,x).predict([.7,.3,-.2]).reshape(6,-1)
    samples=[rng.choice(theta,size=100000,p=d/d.sum()) for d in density]
    fits=[]
    for bins in (48,96):
        centers=-np.pi+(np.arange(bins)+.5)*2*np.pi/bins
        X=np.column_stack((np.tile(centers,6),np.repeat(boundaries,bins,axis=0)))
        freq=np.concatenate([np.histogram(s,bins=bins,range=(-np.pi,np.pi),density=True)[0] for s in samples])
        fits.append(isolated_fit('two_fixed',{'N':2,'alpha':.3,'p0':[.7,.1],'device':'cuda'},X=X,freq=freq)['new'])
    report['paired_bins']={'source':'same synthetic conditional angle samples','bins':[48,96],
        'fits':fits,'beta_change':fits[1]['beta']-fits[0]['beta'],
        'alpha1_change':fits[1]['alpha1']-fits[0]['alpha1'],
        'convergence_claim':False}
    report['wall_seconds']=time.perf_counter()-started
    Path(destination).write_text(json.dumps(report,indent=2))
    print(destination)

if __name__=='__main__': main(sys.argv[1])
