import os,sys,time,json
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
from scipy.linalg import inv
from scipy.optimize import root
T=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(T/'archived_study'));sys.path.insert(0,str(T/'archived_study/deps'))
from run_robustness import setup,SCENARIOS
import run_multiseed as m
from independent_decimal_krawczyk import fpower_network,security_case9
arch=json.load(open(T/'archived_study/results.json'))
SEEDS=[711,713,717]
RAD={'IEEE9':['0.0015','0.003','0.006'],'IEEE14':['0.0025','0.005','0.01']}
rel='0.0000001'; inj='0.000000001'
RESULT=[]
for case in arch['cases']:
    name=case['name'];scenario=case['scenario'];scales={s:(scale,idx) for s,scale,idx in SCENARIOS[name]}
    scale,idx=scales[scenario];d=setup(name,scale,idx)
    oracle=m.Oracle(d['A'],d['b'])
    observed=[]
    for prior in case['trials']:
        seed=prior['seed'];theta=m.train(oracle,seed);z,met=m.holdout(oracle,d,theta,seed)
        center=d['xinit']+z
        recompute_diff=abs(met['actual_linear_residual_inf']-prior['quantum_linear_residual_inf'])
        assert recompute_diff<1e-12,(name,scenario,seed,recompute_diff)
        B=inv(d['J'](center));accepted=False;checked=[]
        for rad in RAD[name]:
            v=fpower_network(name,center,rad,rel,inj,B,load_scale=scale,outage_branch_index=idx)
            q=dict(radius=rad,strict=v['strict_inclusion'],min_margin=v['min_margin'],max_row=v['max_majorant_row_sum'])
            checked.append(q)
            if v['strict_inclusion']:
                accepted=True
                if name=='IEEE9':
                    s=security_case9(center,rad,rel)
                    q['native_voltage_safe']=s['all_voltages_safe'];q['both_end_branches_safe']=s['all_branches_safe'];q['branch_margin_MVA_lower']=s['min_branch_margin_mva']
                break
        original=prior['method_results']['1e-07/VQLS_reconstructed']['accepted']
        trial=dict(seed=seed,archive_accepted=original,independent_accepted=accepted,agreement=original==accepted,checks=checked)
        observed.append(trial)
        print('SCENARIO',name,scenario,'SEED',seed,'OLD',original,'NEW',accepted,'MARGIN',checked[-1]['min_margin'],flush=True)
    # independent classical full nonlinear root probe at radius 1e-4 (base cases) or adaptive radii 1e-4/1e-3
    classical=root(d['F'],d['xinit'],jac=d['J'],options={'xtol':1e-12,'maxfev':1800})
    if np.linalg.norm(d['F'](classical.x),np.inf)<1e-8:
        B=inv(d['J'](classical.x))
        aa=[]
        for r in ['0.0001','0.0003','0.001','0.003']:
            z=fpower_network(name,classical.x,r,rel,inj,B,load_scale=scale,outage_branch_index=idx)
            aa.append(dict(radius=r,inclusion=z['strict_inclusion'],min_margin=z['min_margin']))
            if z['strict_inclusion']:break
    else:aa=[]
    RESULT.append(dict(network=name,scenario=scenario,quantum_trials=observed,classical_root_radius_test=aa))
    T.joinpath('full_scenario_audit_working.json').write_text(json.dumps(RESULT,indent=2))
T.joinpath('full_scenario_audit.json').write_text(json.dumps(RESULT,indent=2))
print('COMPLETE all',len(RESULT),'cases, total',sum(len(x['quantum_trials']) for x in RESULT),'trials',flush=True)
