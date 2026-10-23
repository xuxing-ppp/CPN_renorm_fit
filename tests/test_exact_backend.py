"""Independent density/derivative, recovery and sequential-pipeline regressions."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from cpn_renorm.exact_fitting import Conditional as CPUConditional, fit_two_fixed
from cpn_renorm.fitting import one_plaq_probability
from cpn_renorm.torch_fitting import Conditional, OnePlaquette, fit_one, fit_two_fixed as torch_fit
from cpn_renorm.pipeline import _default_alpha_points, _run_topo, _run_beta1_scan, _run_one_plaq, _run_two_plaq
from cpn_renorm.config import load_config, mod_config
from cpn_renorm.patches import _fine_couplings

ROOT = Path(__file__).parents[1]


def features(bins=48):
    rng = np.random.default_rng(731)
    boundary = rng.uniform(-np.pi,np.pi,(6,8))
    boundary[:,6] = rng.uniform(.15,.9,6)
    theta = -np.pi+(np.arange(bins)+.5)*2*np.pi/bins
    return np.column_stack((np.tile(theta,6),np.repeat(boundary,bins,axis=0)))


class ExactNumericsTests(unittest.TestCase):
    def test_one_probability_and_analytic_derivative(self):
        X = np.array([[s,d] for s in range(-3,4) for d in (-2.8,.1,2.9)])
        for alpha in (.0001,.1,.49,.5,2.,40.):
            pred, jac = one_plaq_probability(X,alpha)
            model = OnePlaquette(X,'cpu')
            actual, derivative = model.evaluate(model.tensor([alpha]))
            np.testing.assert_allclose(actual.numpy(),pred,atol=1e-13,rtol=1e-11)
            np.testing.assert_allclose(derivative[:,0].numpy(),jac,atol=1e-12,rtol=1e-9)
            h = alpha*1e-5
            fd=(one_plaq_probability(X,alpha+h)[0]-one_plaq_probability(X,alpha-h)[0])/(2*h)
            np.testing.assert_allclose(jac,fd,atol=1e-7,rtol=1e-6)
        target = one_plaq_probability(X,.4)[0]
        result = fit_one(X,target,.6,'cpu')
        self.assertAlmostEqual(result['alpha'],.4,places=7)

    def test_exact_density_jacobian_and_sharp_normalization(self):
        X=features()
        for alpha,beta,a1 in ((0.,.4,-.2),(.02,.5,.3),(.3,.8,-.5),(2.,8.,3.)):
            cpu=CPUConditional(2,X,nodes=2048)
            model=Conditional(2,X,nodes=2048,device='cpu')
            p=[beta,alpha,a1]
            pred,jac=model.evaluate(p)
            np.testing.assert_allclose(pred.numpy(),cpu.predict(p),atol=2e-12,rtol=1e-10)
            for index in (0,2):
                lo,hi=np.array(p),np.array(p)
                lo[index]-=1e-5; hi[index]+=1e-5
                fd=(cpu.predict(hi)-cpu.predict(lo))/2e-5
                np.testing.assert_allclose(jac[:,index].numpy(),fd,atol=2e-7,rtol=1e-5)

    def test_fixed_alpha_recovery_and_zero_correspondence(self):
        X=features()
        for alpha,a1 in ((0.,.25),(.3,-.2),(.9,-.5)):
            freq=CPUConditional(2,X).predict([.7,alpha,a1])
            cpu=fit_two_fixed(2,X,freq,alpha)
            gpu=torch_fit(2,X,freq,alpha,device='cpu')
            for result in (cpu,gpu):
                self.assertAlmostEqual(result['new']['beta'],.7,places=6)
                self.assertAlmostEqual(result['new']['alpha1'],a1,places=6)
                self.assertLess(result['new']['mse'],1e-18)
            if alpha==0:
                self.assertAlmostEqual(cpu['old']['alpha_eff'],a1,places=6)
        wrong=fit_two_fixed(2,X,freq,0.)
        self.assertGreater(wrong['new']['mse'],cpu['new']['mse']+1e-8)

    def test_full_fine_action_and_zero_sampling_equivalence(self):
        cfg=mod_config(load_config(ROOT/'configs/example.toml'),1)
        cfg['model'].update(alpha=.3,alpha1=-.2)
        coupling,kind=_fine_couplings(cfg)
        self.assertEqual(kind,'full')
        self.assertEqual((coupling.alpha,coupling.alpha1),(.3,-.2))
        cfg['model']['alpha']=0.
        coupling,kind=_fine_couplings(cfg)
        self.assertEqual(kind,'half')
        self.assertEqual((coupling.alpha,coupling.alpha1),(-.2,0.))

    def test_centered_scan_and_limits(self):
        scan={'min':0.,'max':2.,'points':[],'refine_tolerance':.02,'fit_spacing_min':.01}
        np.testing.assert_allclose(_default_alpha_points(.7,scan),[.68,.7,.72])
        scan['points']=[.1,.4]
        self.assertEqual(_default_alpha_points(.7,scan),[.1,.4])
        with self.assertRaises(ValueError): _default_alpha_points(3.,scan)

    def test_u_zero_subspace_skips_sampling_and_identification(self):
        cfg=mod_config(load_config(ROOT/'configs/example.toml'),1)
        cfg['renormalization']['type']=['U']
        cfg['runtime']['device']='cpu'
        cfg['model']['alpha']=0.
        with tempfile.TemporaryDirectory() as tmp, patch('cpn_renorm.pipeline.generate_one_plaq') as sample:
            result=_run_one_plaq(cfg,Path(tmp),{})
            sample.assert_not_called()
            self.assertEqual(result['fits']['U']['alpha'],0.)
            self.assertIn('convention',result['fits']['U']['source'])

    def test_obs_full_and_topo_keeps_alpha1_fixed(self):
        cfg=mod_config(load_config(ROOT/'configs/example.toml'),1)
        cfg['runtime'].update(device='cpu',auto_chains=False)
        cfg['model'].update(alpha=.2,alpha1=-.3)
        cfg['renormalization']['beta1_methods']=['corrG']
        geometry={'factor':1,'L_fine':8,'L_coarse':8,'padding':0}
        calls=[]
        def measure(_cfg,_dir,_settings,**kw):
            calls.append(kw)
            return {'xi_G':1.,'xi_G_err':.01,'topology':{'Q_z':{'topo_sus':.5,'err':.01}}}
        def search(_cfg,_scan,measured,phases,measure,*args,**kwargs):
            measure(.1)
            return {'value':None,'bracketed':False}
        with tempfile.TemporaryDirectory() as tmp, patch('cpn_renorm.pipeline._measure_point',side_effect=measure), patch('cpn_renorm.pipeline._search_fit',side_effect=search), patch('cpn_renorm.pipeline.isolated_plot'):
            _run_beta1_scan(cfg,Path(tmp),geometry,.7,-.4,alpha_c=.3)
            self.assertEqual((calls[0]['alpha'],calls[0]['alpha1'],calls[0]['kind']),(.2,-.3,'full'))
            self.assertEqual((calls[1]['alpha'],calls[1]['alpha1'],calls[1]['kind']),(.3,-.4,'full'))
            calls.clear()
            result=_run_topo(cfg,Path(tmp),geometry,.7,-.4,{'corrG':{'value':.1,'bracketed':True}},.3)
            self.assertEqual(calls[1]['alpha1'],-.4)
            self.assertEqual(calls[1]['beta'],.7)
            self.assertEqual(calls[1]['beta1'],.1)
            self.assertFalse(result['matches']['corrG']['correction_success'])
            self.assertEqual(result['matches']['corrG']['alpha_after'],.3)

    def test_wrong_fixed_alpha_can_have_higher_mse_than_old_control(self):
        X=features()
        freq=CPUConditional(2,X).predict([.7,0.,.2])
        result=fit_two_fixed(2,X,freq,1.5)
        self.assertGreater(result['new']['mse'],result['old']['mse']+1e-6)
        self.assertEqual(result['new']['alpha'],1.5)

    def test_fixed_alpha_and_backend_invalidate_fit_identity(self):
        from cpn_renorm.pipeline import _two_fit_key, _scan_fingerprint
        from copy import deepcopy
        cfg=mod_config(load_config(ROOT/'configs/example.toml'),1)
        cfg['runtime']['device']='cpu'
        one={'fits':{'z':{'alpha':.2,'alpha_err':.01}}}
        key=_two_fit_key(cfg,{},one)
        one['fits']['z']['alpha']=.3
        self.assertNotEqual(key,_two_fit_key(cfg,{},one))
        for kind in ('alpha','beta1'):
            a=_scan_fingerprint(kind,cfg,{},renorm_type='z',beta_c=.6,alpha_c=.2,alpha1_c=-.3)
            b=_scan_fingerprint(kind,cfg,{},renorm_type='z',beta_c=.6,alpha_c=.3,alpha1_c=-.3)
            self.assertNotEqual(a,b)
        with patch('cpn_renorm.pipeline.resolve_device',return_value='cuda'):
            self.assertNotEqual(key,_two_fit_key(cfg,{},one))

    def test_pipeline_order_updates_outputs_without_archives(self):
        import json
        from cpn_renorm.pipeline import run_pipeline
        cfg=load_config(ROOT/'configs/example.toml')
        cfg['runtime'].update(device='cpu',auto_chains=False)
        cfg['renormalization'].update(beta1_methods=[])
        geometry={'padding':0,'factor':2,'L_fine':8,'L_coarse':4}
        events=[]
        def pilot(*args):
            events.append('pilot');return {'geometry':geometry}
        def one(*args):
            events.append('one');return {'fits':{'z':{'alpha':.2}}}
        def two(*args):
            events.append('two')
            self.assertEqual(args[3]['fits']['z']['alpha'],.2)
            return {'fits':{'z':{'old':{'alpha_eff':.1},'new':{'beta':.6,'alpha1':-.1}}}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'example'; root.mkdir()
            with patch('cpn_renorm.pipeline.run_pilot',side_effect=pilot), patch('cpn_renorm.pipeline._run_one_plaq',side_effect=one), patch('cpn_renorm.pipeline._run_two_plaq',side_effect=two):
                run_pipeline(cfg,tmp)
                self.assertEqual(events,['pilot','one','two'])
                first_summary=json.loads((root/'summary.json').read_text())
                first_resolved=json.loads((root/'resolved.json').read_text())
                run_pipeline(cfg,tmp)
            self.assertEqual(json.loads((root/'summary.json').read_text()),first_summary)
            self.assertEqual(json.loads((root/'resolved.json').read_text()),first_resolved)
            self.assertNotIn('pipeline_revision',first_summary)
            self.assertNotIn('pipeline_revision',first_resolved)
            self.assertFalse(list(root.glob('*.legacy.json')))
            self.assertFalse(list(root.rglob('*.npz')))
