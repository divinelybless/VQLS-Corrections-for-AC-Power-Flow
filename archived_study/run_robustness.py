"""Fixed-budget load and N-1 robustness study, using archived full-network VQLS sampler.
Independent classical interval Krawczyk around measured solution centers.
NOT Qiskit/Aer; no device noise; no claim of end-to-end quantum performance.
"""
import os,sys,json,time,math
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.optimize import root
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from mpmath import iv
HERE=Path(__file__).resolve().parent
DEPS=HERE/'deps'
sys.path.insert(0,str(DEPS))
from prior.verify_power import cases,model
import run_multiseed as run
from interval_center_test import run_case,ends,point,absub
from certify import RATES9
run.STEPS=120
SEEDS=(711,713,717)
LEVELS=(1e-7,1e-6)
RADIUS={'IEEE9':(0.0015,0.003,0.006),'IEEE14':(0.0025,0.005,0.010)}
# Index of omitted line in the bundled MATPOWER branch ordering, None = intact.
SCENARIOS={
 'IEEE9':[('base',1.0,None),('load20',1.20,None),('outage_4_5',1.0,1),('load10_outage_9_4',1.10,8)],
 'IEEE14':[('base',1.0,None),('load20',1.20,None),('outage_1_2',1.0,0),('load10_outage_5_6',1.10,9)]}

def setup(name,scale,out):
    bus,edges,vm,pg=cases()[name]
    bus=bus.copy();edges=edges.copy();bus[:,1]*=scale;bus[:,2]*=scale
    old_idx=np.arange(len(edges))
    if out is not None:
        edges=np.delete(edges,out,axis=0);old_idx=np.delete(old_idx,out)
    Y,ang,pq,unpack,F,J,x0=model(bus,edges,vm,pg)
    ref=root(F,x0,jac=J,options={'maxfev':2500,'xtol':1e-11})
    ref_ok=bool(np.isfinite(ref.x).all() and np.linalg.norm(F(ref.x),np.inf)<1e-8)
    # Local warm-start at known base-case solution, NOT a scenario-specific root.
    base_bus,base_edge,base_vm,base_pg=cases()[name]
    _,_,_,_,base_F,base_J,base_x0=model(base_bus,base_edge,base_vm,base_pg)
    baseroot=root(base_F,base_x0,jac=base_J,tol=1e-12)
    assert np.linalg.norm(base_F(baseroot.x),np.inf)<1e-9
    xinit=baseroot.x+0.0012*np.sin(np.arange(len(baseroot.x))*.73+.5)
    d={'name':name,'bus':bus,'edges':edges,'old_idx':old_idx,'vm':vm,'pg':pg,
       'F':F,'J':J,'xinit':xinit,'unpack':unpack,'A':J(xinit),'b':-F(xinit),
       'reference_root_found':ref_ok,'ref':ref.x if ref_ok else None,'sol':ref.x,'nominal_root_residual':float(np.linalg.norm(F(ref.x),np.inf))}
    return d

def limits(name,d,center,rad,rel):
    bus=d['bus'];edge=d['edges'];ang=np.flatnonzero(bus[:,0]!=3);pq=np.flatnonzero(bus[:,0]==1);n=len(bus)
    th=[point(v) for v in np.deg2rad(bus[:,6])]
    vv=[point(v) for v in bus[:,5]]
    for k,v in d['vm'].items():vv[k-1]=point(v)
    for k,i in enumerate(ang):th[i]=iv.mpf([float(center[k]-rad),float(center[k]+rad)])
    for k,i in enumerate(pq):vv[i]=iv.mpf([float(center[len(ang)+k]-rad),float(center[len(ang)+k]+rad)])
    v_lo=min(ends(v)[0] for v in vv);v_hi=max(ends(v)[1] for v in vv)
    v_native=(0.9,1.1) if name=='IEEE9' else (0.94,1.06)
    voltage_native=bool(v_lo>=v_native[0] and v_hi<=v_native[1]); voltage_illustrative=bool(v_lo>=.9 and v_hi<=1.1)
    bad_native=[int(i+1) for i,v in enumerate(vv) if ends(v)[0]>v_native[1] or ends(v)[1]<v_native[0]]
    result=dict(voltage_native=voltage_native,voltage_illustrative=voltage_illustrative,native_violated_buses=bad_native,vmin_lower=float(v_lo),vmax_upper=float(v_hi))
    if name=='IEEE14':
        result['branch_rating_available']=False
        result['branch_safe']=None
        return result
    def prim(z):
        q=float(z);eps=abs(q)*rel if q else 0
        return iv.mpf([q-eps,q+eps])
    def mul(g,h,x,y):return g*x-h*y,g*y+h*x
    margins=[]
    for row,idx in zip(edge,d['old_idx']):
        f,t,rr,xx,bb,tap=row; i=int(f)-1;j=int(t)-1
        rr,xx,bb=prim(rr),prim(xx),prim(bb);a=prim(tap) if tap else point(1)
        den=rr*rr+xx*xx;g=rr/den;h=-xx/den
        vi_r,vi_i=vv[i]*iv.cos(th[i]),vv[i]*iv.sin(th[i]);vj_r,vj_i=vv[j]*iv.cos(th[j]),vv[j]*iv.sin(th[j])
        a1,b1=mul(g/a**2,(h+bb/2)/a**2,vi_r,vi_i)
        a2,b2=mul(-g/a,-h/a,vj_r,vj_i)
        ar,ai=a1+a2,b1+b2
        a3,b3=mul(g,h+bb/2,vj_r,vj_i);a4,b4=mul(-g/a,-h/a,vi_r,vi_i)
        br,bi=a3+a4,b3+b4
        def upper(vr,vi,ir,ii):
            sr=vr*ir+vi*ii;si=vi*ir-vr*ii
            return 100*np.hypot(absub(sr),absub(si))
        margins.append(float(RATES9[idx]-max(upper(vi_r,vi_i,ar,ai),upper(vj_r,vj_i,br,bi))))
    result['branch_rating_available']=True
    result['branch_min_margin_MVA']=min(margins)
    result['branch_safe']=bool(min(margins)>=0)
    return result

def certify(name,d,z,rel):
    ctr=d['xinit']+z
    results=[]
    for r in RADIUS[name]:
        p=run_case(name,d['bus'],d['edges'],d['vm'],d['pg'],rel=rel,inject=rel*.01,radius=r,center=ctr)
        k=bool(p.get('krawczyk_strict_inclusion',False))
        lim=limits(name,d,ctr,r,rel) if k else None
        results.append(dict(radius=r,root_enclosure=k,limit_evaluation=lim,
          interval_majorant_inf=p.get('interval_majorant_rowsum'),
          inclusion_margin=p.get('krawczyk_inclusion_margin_min')))
        if k:break
    success=next((p for p in results if p['root_enclosure']),None)
    return dict(accepted=bool(success),min_radius=(success['radius'] if success else None),
         native_safe=(bool(success['limit_evaluation']['voltage_native']) and (name!='IEEE9' or bool(success['limit_evaluation']['branch_safe']))) if success else None,
         tests=results)

def compute():
    out={'study':'Multi-condition fixed-shot VQLS candidate and independent interval fallback robustness',
      'date':'2026-11-02',
      'methods':'Ideal-outcome Pauli-sampling, no installed Qiskit/Aer, mpmath.iv interval check not rounding audited',
      'seeds':list(SEEDS),'levels':list(LEVELS),'radii':RADIUS,
      'fairness':'Same scenario, same initial warm-start and interval uncertainties for all correction methods. Full Newton is a practical classical baseline; damped LU is diagnostic.',
      'limitations':['Base-case solved state used as common warm-start for all scenarios; not cold-start realistic deployment.','A single quantum Newton correction is tested, not a complete iterative quantum solver.','IEEE 14 native voltage limits can be violated by generator setpoints; absent thermal limits.','Classical direct solve of baseline state is part of initialization cost; no speedup claims.'], 'cases':[]}
    for name,scenarios in SCENARIOS.items():
      for scenario,scale,out_idx in scenarios:
        d=setup(name,scale,out_idx)
        b=d['b'];A=d['A'];lu=splu(csc_matrix(A)).solve(b)
        if d['reference_root_found']:
            root_err=float(np.linalg.norm(d['xinit']+lu-d['ref'],np.inf))
        else:root_err=None
        item=dict(name=name,scenario=scenario,load_scale=scale,outage_branch_index=out_idx,
                  outage_endpoints=None if out_idx is None else list(map(int,cases()[name][1][out_idx,:2])),
                  nstate=len(b),initial_mismatch_inf=float(np.linalg.norm(b,np.inf)),
                  jacobian_cond_warmstart=float(np.linalg.cond(A)),
                  root_found=d['reference_root_found'], root_error_one_step_lu=root_err, trials=[])
        # Realistic classical full nonlinear reference with same warmstart, but no reliance for training.
        cold=root(d['F'],d['xinit'],jac=d['J'],options={'maxfev':2200,'xtol':1e-11})
        item['classical_full_newton_converged']=bool(np.linalg.norm(d['F'](cold.x),np.inf)<1e-8)
        item['classical_full_newton_residual_inf']=float(np.linalg.norm(d['F'](cold.x),np.inf))
        oracle=run.Oracle(A,b)
        q_results=[]
        for seed in SEEDS:
          oracle.train_calls=0;oracle.train_shots=0
          tic=time.perf_counter();theta=run.train(oracle,seed)
          z,met=run.holdout(oracle,d,theta,seed)
          target=np.linalg.norm(A@z-b,np.inf)
          damping=min(1.0,target/max(np.linalg.norm(b,np.inf),1e-14))
          matched=(1-damping)*lu
          e=dict(seed=seed,quantum_linear_residual_inf=float(np.linalg.norm(A@z-b,np.inf)),
             matched_lu_linear_residual_inf=float(np.linalg.norm(A@matched-b,np.inf)),
             full_lu_linear_residual_inf=float(np.linalg.norm(A@lu-b,np.inf)),
             quantum_nonlinear_mismatch_inf=float(np.linalg.norm(d['F'](d['xinit']+z),np.inf)),
             classical_matched_nonlinear_mismatch_inf=float(np.linalg.norm(d['F'](d['xinit']+matched),np.inf)),
             ideal_projective_loss=float(met['true_projective_loss_diagnostic']),
             measurement_residual_upper=float(met['certified_linear_residual_bound_inf']),
             measurement_scale_confidence_interval_possible=met['scale_CI_possible'],
             theoretical_joint_confidence=0.95,shots=int(met['shots_total']),
             oracle_imaginary_pauli_coefficients=int(met['oracle_imaginary_coeff_terms']),
             method_results={},elapsed_train_and_holdout=float(time.perf_counter()-tic))
          for level in LEVELS:
             for method,correction in [('VQLS_reconstructed',z),('matched_damped_LU',matched),('single_LU_Newton',lu),('full_classical_Newton',cold.x-d['xinit'])]:
                 if method=='full_classical_Newton' and not item['classical_full_newton_converged']:continue
                 c=certify(name,d,correction,level)
                 e['method_results'][f'{level:g}/{method}']=c
          item['trials'].append(e)
          print(name,scenario,seed,'linres',round(e['quantum_linear_residual_inf'],5),'low-unc cert',e['method_results'].get('1e-07/VQLS_reconstructed',{}).get('accepted'),'shots',e['shots'],flush=True)
        out['cases'].append(item)
        HERE.joinpath('results_working.json').write_text(json.dumps(out,indent=2))
    HERE.joinpath('results.json').write_text(json.dumps(out,indent=2))
    print('DONE',flush=True)
if __name__=='__main__':compute()
