"""Controlled certificate-acceptance and cost comparison.
Uses the archived IEEE 9-bus finite-shot VQLS candidate; independently trains a
five-state-qubit finite-shot VQLS for 14-bus in an ideal-outcome simulator.
No physical QPU or Qiskit execution. No exact inverse solution used by training.
Classical matched comparator uses an LU solve and intentional residual matching,
so it is diagnostic, not an efficient standalone algorithm.
"""
import os, json, sys, time, itertools, shutil
from pathlib import Path
import numpy as np
from scipy.optimize import root
from scipy.linalg import solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu, gmres
BASE=Path(__file__).resolve().parent
PRIOR=BASE
sys.path.insert(0,str(PRIOR))
from prior.verify_power import cases, model
from experiment_full_vqls import prep as prep4, angles_from_known_rhs as init4
from prior.run_experiment import PAULI

def prep(theta):
    vec=np.array([1.]); i=0
    while i<len(theta):
        n=len(vec)
        if i+n>len(theta): raise ValueError('inconsistent tree')
        v=np.empty(2*n)
        v[0::2]=vec*np.cos(theta[i:i+n]);v[1::2]=vec*np.sin(theta[i:i+n]);vec=v;i+=n
    return vec

def angles_from_rhs(y):
    out=[];nodes=[np.asarray(y,dtype=float)]
    depth=int(np.log2(len(y)))
    for dep in range(depth):
        nxt=[]
        for v in nodes:
            a,b=np.split(v,2)
            t=np.arctan2(np.linalg.norm(b),np.linalg.norm(a)) if dep<depth-1 else np.arctan2(b[0],a[0])
            out.append(t);nxt.extend([a,b])
        nodes=nxt
    return np.array(out)

def decomp(A):
    n=int(round(np.log2(len(A))))
    P=[];word=[]
    for key in itertools.product('IXYZ',repeat=n):
        M=np.array([[1]],complex)
        for k in key:M=np.kron(M,PAULI[k])
        c=np.einsum('ij,ji->',M,A)/len(A)
        if abs(c)>1e-11:
            P.append(M); word.append((int(len(word)),''.join(key),float(c.real)))
    # second enumeration for coefficients to avoid mistaken indices
    C=np.array([np.einsum('ij,ji->',P_i,A).real/len(A) for P_i in P])
    return C,np.asarray(P),len(word)

class ShotDriven:
    def __init__(self,J,b,seed=20261102):
        self.rng=np.random.default_rng(seed)
        self.m=len(b);self.pow=2**int(np.ceil(np.log2(self.m)))
        self.M=np.zeros((self.pow,self.pow));self.M[:self.m,:self.m]=J
        self.jnorm=np.linalg.norm(J,2); self.M/=self.jnorm
        self.K=self.M.T@self.M
        self.u=np.zeros(self.pow); self.u[:self.m]=b/np.linalg.norm(b)
        self.ca,self.Pa,self.npa=decomp(self.M)
        self.cd,self.Pd,self.npd=decomp(self.K)
        self.Ga=float(np.sum(abs(self.ca)));self.Gd=float(np.sum(abs(self.cd)))
        self.eval_count=0;self.shots=0
    def sample(self,theta,kind,n):
        v=prep(theta)
        if kind=='a':
            c,P,scale=self.ca,self.Pa,self.Ga
            means=np.real(np.einsum('i,tij,j->t',self.u,P,v)*(np.sign(c)))
        else:
            c,P,scale=self.cd,self.Pd,self.Gd
            means=np.real(np.einsum('i,tij,j->t',v,P,v)*(np.sign(c)))
        probs=abs(c)/scale
        picks=self.rng.multinomial(n,probs)
        plus=self.rng.binomial(picks,np.clip((1+means)/2,0,1))
        return float(scale*(2*plus.sum()-n)/n)
    def cost(self,theta,n=10000):
        a=self.sample(theta,'a',n);D=self.sample(theta,'D',n)
        self.eval_count+=1;self.shots+=2*n
        return float(np.clip(1-a*a/max(D,.08),-1.5,2)),a,D
    def train(self,steps=250,seed=77,shots=10000):
        theta=angles_from_rhs(self.u)
        assert np.linalg.norm(prep(theta)-self.u)<1e-12
        rg=np.random.default_rng(seed);checks=[]
        start=time.perf_counter()
        for k in range(1,steps+1):
            ck=.20/(k**.101);ak=.055/(k+12)**.602
            delta=rg.choice([-1.,1.],size=len(theta))
            fp,_,_=self.cost(theta+ck*delta,shots)
            fm,_,_=self.cost(theta-ck*delta,shots)
            theta=np.mod(theta-ak*np.clip((fp-fm)/(2*ck)*delta,-6.,6.)+np.pi,2*np.pi)-np.pi
            if k in [1,20,50,100,150,200,steps]:
                loss,a,D=self.cost(theta,5*shots)
                checks.append({'iteration':k,'sampled_loss':loss,'a':a,'D':D})
        elapsed=time.perf_counter()-start
        return theta,checks,elapsed

def get_case(name):
    bus,edge,vm,pg=cases()[name]
    _,ang,pq,unpack,F,J,x0=model(bus,edge,vm,pg)
    sol=root(F,x0,jac=J,tol=1e-12)
    assert np.linalg.norm(F(sol.x),np.inf)<1e-9
    n=len(sol.x);xinit=sol.x + 1.2e-3*np.sin(np.arange(n)*.73+.5)
    return {'name':name,'bus':bus,'edge':edge,'vm':vm,'pg':pg,'ang':ang,'pq':pq,'unpack':unpack,
            'F':F,'J':J,'sol':sol.x,'xinit':xinit,'A':J(xinit),'b':-F(xinit)}

def quantum_candidate9(d):
    rr=json.loads((PRIOR/'results.json').read_text())
    q=np.array(rr['holdout']['reconstructed_delta'])
    assert np.linalg.norm((d['xinit']+q)-np.array(rr['holdout']['xcandidate']),np.inf)<1e-11
    return q,{'system_qubits':5,'train_shots':rr['holdout']['training_shots'],
        'measurement_shots':rr['holdout']['holdout_shots'],'total_shots':rr['holdout']['total_allocation_shots'],
        'evaluations':rr['holdout']['training_objective_evaluations'],
        'training_method':'archived 5-qubit SPSA measurement-driven VQLS with independently sampled holdout',
        'noiseless_vqls_objective_diagnostic':rr['loss_final_diagnostic'],
        'confidence_residual_upper':rr['holdout']['max_certified_linear_residual_upper']}

def quantum_candidate14(d):
    o=ShotDriven(d['A'],d['b'],seed=20261102)
    th,checks,secs=o.train(steps=250)
    # Independent finite-shot measurements of measurable a, D and real-amplitude overlaps
    rg=np.random.default_rng(77814)
    v=prep(th); nscale=np.linalg.norm(d['b'])/o.jnorm
    a=o.sample(th,'a',800000);D=o.sample(th,'D',800000)
    scale=float(nscale*a/D) if D>0 else float('nan')
    n_v=120000
    cnt=rg.binomial(n_v,np.clip((1+v[:len(d['b'])])/2,0,1))
    vh=2*cnt/n_v-1
    q=scale*vh
    hold_shots=1600000+n_v*len(d['b'])
    # The actual holdout does not estimate all residuals. Certificate acceptance is purely classical interval.
    info={'system_qubits':6,'train_shots':o.shots,'measurement_shots':hold_shots,
      'total_shots':o.shots+hold_shots,'evaluations':o.eval_count,
      'training_method':'new 6-qubit ideal-outcome sampled-SPSA, independent a,D, amplitude readout',
      'noiseless_vqls_objective_diagnostic':float(1-(o.u@(o.M@v))**2/((o.M@v)@(o.M@v))),
      'measured_scale':scale,'shot_holdout_a':a,'shot_holdout_D':D,
      'measurement_scale_width_abs_bound':float(nscale*o.Ga*np.sqrt(2*np.log(8/.05)/800000)),
      'sampler_training_seconds':secs,'checkpoint_loss':checks,
      'readout_amplitude_width':float(np.sqrt(2*np.log(2*len(d['b'])/.0125)/n_v)),
      'confidence_residual_upper':None,
      'warning':'No separately measured 22-coordinate quantum residual; physical certificate is a classical interval Krawczyk check of reconstructed correction.'}
    return q,info

def metrics(d,q):
    x=d['xinit']+q
    return {'linear_residual_inf':float(np.linalg.norm(d['A']@q-d['b'],np.inf)),
      'nonlinear_residual_inf':float(np.linalg.norm(d['F'](x),np.inf)),
      'state_error_inf_diagnostic':float(np.linalg.norm(x-d['sol'],np.inf)),
      'voltage_min_point':float(np.min(d['unpack'](x)[1])),
      'voltage_max_point':float(np.max(d['unpack'](x)[1])),
      'is_point_voltage_safe_0p9_1p1':bool(np.all((d['unpack'](x)[1]>=.9)&(d['unpack'](x)[1]<=1.1)))}

def time_lu(d,repeats=300):
    A=d['A'];b=d['b']; ms=[]
    for k in range(repeats):
        t=time.perf_counter();u=splu(csc_matrix(A));v=u.solve(b);ms.append(time.perf_counter()-t)
    return float(np.median(ms)),v

def build_candidates(d,q):
    A=d['A'];b=d['b']
    t,full=time_lu(d)
    target=np.linalg.norm(A@q-b,np.inf)
    initial=np.linalg.norm(b,np.inf)
    damping=float(min(1.,max(0.,target/initial))) if initial else 0
    matched=(1-damping)*full
    # Alternate inexact iterative method, natural GMRES stopping on comparable relative tolerance
    n=len(b)
    counter=[0]
    def cb(r):counter[0]+=1
    gm,code=gmres(csc_matrix(A),b,rtol=min(.95,max(target/np.linalg.norm(b),1e-8)),atol=0,callback=cb,callback_type='pr_norm',restart=n,maxiter=1)
    return {'quantum_reconstructed':q,'classical_matched_damped_LU':matched,'classical_full_sparse_LU':full,
            'classical_GMRES_natural':gm}, {'classical_LU_median_seconds':t,
            'match_target_linear_inf':float(target),'classical_matched_damping':damping,
            'GMRES_status':int(code),'GMRES_iterations':int(counter[0])}

def main():
    result={'description':'Controlled certificate acceptance vs total resource use; no quantum hardware execution',
    'uncertainty_rel_primitive':1e-7,'uncertainty_injection_pu':1e-9,
    'comparison_note':'Classical matched case uses LU then deliberate damping; comparable residual, not a free approximate iterative solve.',
    'certification_note':'Krawczyk inclusion is classical for realized centre; measured-only state bound is not implied.',
    'cases':{}}
    for name in ['IEEE9','IEEE14']:
        d=get_case(name)
        if name=='IEEE9':q,info=quantum_candidate9(d)
        else:q,info=quantum_candidate14(d)
        if not np.isfinite(q).all(): raise RuntimeError('nonfinite quantum reconstruction')
        cand,cost=build_candidates(d,q)
        rows={k:metrics(d,v) for k,v in cand.items()}
        result['cases'][name]={'n_states':len(d['b']),'condition_2':float(np.linalg.cond(d['A'])),
            'quantum':info,'classical':cost,'candidates':rows,
            'candidate_vectors':{k:v.tolist() for k,v in cand.items()},
            'centres':{k:(d['xinit']+v).tolist() for k,v in cand.items()}}
        print(name, 'quantum shots',info['total_shots'], 'sampler runtime',info.get('sampler_training_seconds'),flush=True)
        for k,v in rows.items():print(' ',k, 'linear',v['linear_residual_inf'],'nonlinear',v['nonlinear_residual_inf'],'state_error',v['state_error_inf_diagnostic'],flush=True)
    (BASE/'measurement_and_matching_results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
