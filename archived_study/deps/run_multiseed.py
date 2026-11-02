"""Fixed-budget, independent finite-shot VQLS multi-seed experiment.
No Qiskit/Aer in execution environment. Training samples *ideal* Pauli/Hadamard
observable outcomes, with correct complex Pauli coefficients for nonsymmetric J.
All full-network certificates are independently checked by prior interval code.
"""
import os,sys,json,time,itertools,math
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
BASE=Path(__file__).resolve().parent
OLD=BASE  # portable package: supporting modules live beside this script
sys.path.insert(0,str(OLD))
from compare import get_case,angles_from_rhs,prep
from prior.verify_power import cases
from interval_center_test import run_case
from certify import security
from prior.run_experiment import PAULI

SEEDS=list(range(711,719))
STEPS=120
SHOTS_OBJ=4000
SHOTS_SCALE=100000
SHOTS_AMP=10000
SHOTS_RES=10000
ALPHA=.05
RADII={'IEEE9':[0.0005,0.0010,0.0015,0.0025,0.004],
       'IEEE14':[0.0005,0.001,0.002,0.0025,0.004]}

def pauli_decomp(A):
    d=len(A); k=int(round(np.log2(d)))
    words=[];coeff=[];mat=[]
    for key in itertools.product('IXYZ',repeat=k):
        P=np.array([[1+0j]])
        for name in key:P=np.kron(P,PAULI[name])
        c=np.einsum('ij,ji->',P,A)/d
        if abs(c)>1e-10:
            words.append(''.join(key));coeff.append(c);mat.append(P)
    coefficients=np.asarray(coeff,dtype=complex)
    Ps=np.asarray(mat,dtype=complex)
    assert np.allclose(np.einsum('t,tij->ij',coefficients,Ps),A,atol=1e-9)
    return words,coefficients,Ps

class Oracle:
    def __init__(self,J,b):
        self.m=len(b);self.p=2**int(np.ceil(np.log2(self.m)))
        self.jnorm=float(np.linalg.norm(J,2))
        self.M=np.zeros((self.p,self.p));self.M[:self.m,:self.m]=J/self.jnorm
        self.K=self.M.T@self.M
        self.u=np.zeros(self.p);self.u[:self.m]=b/np.linalg.norm(b)
        self.ca_words,self.ca,self.Pa=pauli_decomp(self.M)
        self.cd_words,self.cd,self.Pd=pauli_decomp(self.K)
        self.Ga=float(abs(self.ca).sum());self.Gd=float(abs(self.cd).sum())
        self.train_calls=0;self.train_shots=0
    def sample(self, v, kind, n, rng):
        if kind=='a':
            c=self.ca;P=self.Pa;gamma=self.Ga
            overlap=np.einsum('i,tij,j->t',self.u,P,v,optimize=True)
        else:
            c=self.cd;P=self.Pd;gamma=self.Gd
            overlap=np.einsum('i,tij,j->t',v,P,v,optimize=True)
        mu=np.real(c/abs(c)*overlap)
        assert np.max(abs(mu)) < 1+1e-8
        counts=rng.multinomial(n,np.abs(c)/gamma)
        plus=rng.binomial(counts,np.clip((mu+1)/2,0,1))
        return float(gamma*(2*plus.sum()-n)/n)
    def objective(self, theta, rng):
        v=prep(theta)
        a=self.sample(v,'a',SHOTS_OBJ,rng)
        D=self.sample(v,'D',SHOTS_OBJ,rng)
        self.train_calls+=1;self.train_shots+=2*SHOTS_OBJ
        return float(1-a*a/max(D,0.08))

def train(o,seed):
    rng=np.random.default_rng(seed)
    theta=angles_from_rhs(o.u)
    for k in range(1,STEPS+1):
        ck=.20/(k**.101);ak=.055/(k+12)**.602
        delta=rng.choice([-1.,1.],size=len(theta))
        fp=o.objective(theta+ck*delta,rng)
        fm=o.objective(theta-ck*delta,rng)
        theta=(theta-ak*np.clip((fp-fm)/(2*ck)*delta,-6,6)+np.pi)%(2*np.pi)-np.pi
    return theta

def sample_residual(o,v,rng):
    """Independent importance sampled Pauli overlap estimator of each (J v)_a.
    Measured on a REAL-J basis using complex Pauli expansion (Y terms included).
    """
    c=o.ca;P=o.Pa; gam=o.Ga
    phase=c/abs(c)
    pv=np.einsum('tij,j->ti',P,v,optimize=True)
    mu=np.real(phase[:,None]*pv[:,:o.m]);prob=np.abs(c)/gam
    output=[]
    for i in range(o.m):
        counts=rng.multinomial(SHOTS_RES,prob)
        plus=rng.binomial(counts,np.clip((mu[:,i]+1)/2,0,1))
        output.append(o.jnorm*gam*(2*plus.sum()-SHOTS_RES)/SHOTS_RES)
    ew=o.jnorm*gam*np.sqrt(2*np.log(2*o.m/(ALPHA/3))/SHOTS_RES)
    return np.array(output),float(ew)

def holdout(o,d,theta,seed):
    rng=np.random.default_rng(seed+30933)
    v=prep(theta)
    a=o.sample(v,'a',SHOTS_SCALE,rng);D=o.sample(v,'D',SHOTS_SCALE,rng)
    scale=float(np.linalg.norm(d['b'])/o.jnorm*a/D) if D>0 else np.nan
    cnt=rng.binomial(SHOTS_AMP,np.clip((1+v[:o.m])/2,0,1))
    vh=2*cnt/SHOTS_AMP-1
    w,ew=sample_residual(o,v,rng)
    residual_est=scale*w-d['b']
    e_amp=np.sqrt(2*np.log(2*o.m/(ALPHA/3))/SHOTS_AMP)
    # holds for the realized scale, including selection on the holdout record
    radius=np.abs(residual_est)+abs(scale)*(ew+np.abs(d['A']).sum(axis=1)*e_amp)
    z=scale*vh
    e_scale_a=o.Ga*np.sqrt(2*np.log(2/(ALPHA/6))/SHOTS_SCALE)
    e_scale_D=o.Gd*np.sqrt(2*np.log(2/(ALPHA/6))/SHOTS_SCALE)
    scale_CI_possible=bool(D>e_scale_D)
    resid_meas_correct=bool(np.all(np.abs(w-(d['A']@v[:o.m]))<=ew+1e-12))
    ret=dict(seed=seed,train_shots=o.train_shots,train_calls=o.train_calls,
        holdout_shots=int(2*SHOTS_SCALE+2*o.m*SHOTS_AMP),
        shots_total=int(o.train_shots+2*SHOTS_SCALE+2*o.m*SHOTS_AMP),
        norm_scale=scale,estimate_a=a,estimate_D=D,lower_D=float(D-e_scale_D),scale_CI_possible=scale_CI_possible,
        oracle_pauli_terms=len(o.ca),oracle_pauli_l1=o.Ga,oracle_imaginary_coeff_terms=int(np.sum(np.abs(o.ca.imag)>1e-10)),
        residual_meas_bound=float(ew),residual_measurement_event=resid_meas_correct,
        certified_linear_residual_bound_inf=float(np.max(radius)),
        actual_linear_residual_inf=float(np.linalg.norm(d['A']@z-d['b'],np.inf)),
        actual_nonlinear_residual_inf=float(np.linalg.norm(d['F'](d['xinit']+z),np.inf)),
        state_error_reference_inf=float(np.linalg.norm(d['xinit']+z-d['sol'],np.inf)),
        true_projective_loss_diagnostic=float(1-(o.u@(o.M@v))**2 / np.linalg.norm(o.M@v)**2),
        confidence_joint_nominal=1-ALPHA,
        quantum_circuit_executed=False,
        holdout_sampling='sampled ideal outcome probabilities; no Qiskit/Aer runtime installed')
    return z,ret

def verify(name,d,z):
    bus,edge,vm,pg=cases()[name]
    center=d['xinit']+z
    checks=[]
    for rad in RADII[name]:
        q=run_case(name,bus,edge,vm,pg,rel=1e-7,inject=1e-9,radius=rad,center=center)
        sec=security(name,center,rad) if q['krawczyk_strict_inclusion'] else None
        checks.append(dict(radius=rad,krawczyk_inclusion=bool(q['krawczyk_strict_inclusion']),
              voltage_illustrative=bool(sec['voltage_all_safe_0p9_1p1']) if sec else False,
              voltage_native=bool(sec['native_voltage_all_safe']) if sec else False,
              line_limit=bool(sec['line_security']) if (sec and name=='IEEE9') else None,
              row_norm=q['interval_majorant_rowsum']))
    ok=[x for x in checks if x['krawczyk_inclusion'] and x['voltage_illustrative'] and (name!='IEEE9' or x['line_limit'])]
    return bool(ok),min((x['radius'] for x in ok),default=None),checks

def main():
    allres={'study':'IEEE9/14 fixed-budget multi-seed VQLS sampling & classical interval enclosure',
      'policy':dict(seeds=SEEDS,iterations=STEPS,shots_per_observable_evaluation=SHOTS_OBJ,
          scale_shots_each=SHOTS_SCALE,amplitude_shots_per_coordinate=SHOTS_AMP,
          residual_shots_per_coordinate=SHOTS_RES,confidence=1-ALPHA,
          parameter_uncertainty_relative=1e-7, injection_uncertainty_pu=1e-9, tested_radii=RADII),
      'IMPORTANT':['Statistical holdout bounds are wide; successful interval Krawczyk on classically reconstructed centre is a separate certificate.','Qiskit is not installed; sampled ideal outcomes, not compiled or executed Qiskit circuits.','Voltage constraints for IEEE14 are illustrative; native upper max 1.06 can already be violated by PV setpoints.'], 'networks':{}}
    for name in ['IEEE9','IEEE14']:
        d=get_case(name);oracle=Oracle(d['A'],d['b'])
        lu=splu(csc_matrix(d['A'])).solve(d['b'])
        luok,lurad,luchecks=verify(name,d,lu)
        entries=[]
        for seed in SEEDS:
            t0=time.perf_counter()
            oracle.train_calls=0;oracle.train_shots=0
            theta=train(oracle,seed)
            z,met=holdout(oracle,d,theta,seed)
            ok,minrad,checks=verify(name,d,z)
            # intentionally LU-derived residual-matched comparator (diagnostic only)
            alpha_match=min(1.,met['actual_linear_residual_inf']/np.linalg.norm(d['b'],np.inf))
            z_match=(1-alpha_match)*lu
            mok,mrad,mcheck=verify(name,d,z_match)
            met.update(accepted_interval=ok,min_accepted_radius=minrad,interval_checks=checks,
                matched_LU_accepted=mok, matched_LU_min_radius=mrad,
                matched_LU_linear_residual_inf=float(np.linalg.norm(d['A']@z_match-d['b'],np.inf)),
                elapsed_cpu_sec=float(time.perf_counter()-t0))
            entries.append(met)
            print(name,seed,'projective',round(met['true_projective_loss_diagnostic'],4),'rlin',round(met['actual_linear_residual_inf'],5),'cert',ok,minrad,'matched',mok,mrad,'seconds',round(met['elapsed_cpu_sec'],1),flush=True)
        allres['networks'][name]={'state_dim':len(d['b']),'condition_number':float(np.linalg.cond(d['A'])),
           'classical_LU_certified':luok,'classical_LU_min_radius':lurad,
           'training_pauli_terms':len(oracle.ca),'nonzero_imaginary_pauli_terms':int(np.sum(abs(oracle.ca.imag)>1e-10)),
           'trials':entries,'quantum_interval_acceptance':int(sum(v['accepted_interval'] for v in entries)),
           'matched_classical_acceptance':int(sum(v['matched_LU_accepted'] for v in entries))}
    (BASE/'multiseed_results.json').write_text(json.dumps(allres,indent=2))
    print('DONE',[(k,v['quantum_interval_acceptance'],v['matched_classical_acceptance']) for k,v in allres['networks'].items()],flush=True)

if __name__=='__main__':main()
