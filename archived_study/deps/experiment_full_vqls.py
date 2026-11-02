"""Five-qubit (flag + four tree state-preparation qubits), finite-shot VQLS proof of concept.
Training NEVER reads the exact Newton correction; ideal probabilities only simulate measurement outcomes.
Independent holdout measurements estimate a, D, residual Jv and amplitudes v.
Result is a numerical circuit-outcome simulator, not Qiskit executed nor formally interval certified.
"""
from __future__ import annotations
import numpy as np, scipy.linalg as la, json, time
from pathlib import Path
from scipy.optimize import minimize
from prior.run_experiment import build_full, PAULI
HERE=Path(__file__).parent

# Four-qubit amplitude-tree state: exactly represent arbitrary real 16-d vectors.
# Each tree node corresponds to one multiplexed controlled-Ry rotation.
def prep(theta):
    amp=np.array([1.0])
    idx=0
    for depth in range(4):
        count=2**depth; aa=np.empty(2*count)
        aa[0::2]=amp*np.cos(theta[idx:idx+count]);aa[1::2]=amp*np.sin(theta[idx:idx+count]);amp=aa;idx+=count
    return amp

def angles_from_known_rhs(y):
    # RHS data only -- never an inverse system solution.
    out=[]; nodes=[np.array(y,float)]
    for dep in range(4):
        new=[]
        for v in nodes:
            a,b=np.split(v,2)
            if dep<3:
                theta=np.arctan2(np.linalg.norm(b),np.linalg.norm(a))
                new.extend([a,b])
            else:
                theta=np.arctan2(b[0],a[0]);new.extend([a,b])
            out.append(theta)
        nodes=new
    return np.array(out)

def pauli_decomp(A, tol=1e-11):
    import itertools
    mats=[];cs=[];names=[]
    for key in itertools.product('IXYZ',repeat=4):
        t=''.join(key); M=np.array([[1]],complex)
        for k in t:M=np.kron(M,PAULI[k])
        c=np.trace(M@A)/16
        if abs(c)>tol: names.append(t);cs.append(c);mats.append(M)
    P=np.array(mats); c=np.array(cs)
    assert np.allclose(np.einsum('t,tij->ij',c,P),A,atol=1e-10)
    return names,c,P

class SampledOracle:
 def __init__(self,J,u,seed=20261102):
    self.rng=np.random.default_rng(seed)
    self.J=np.array(J,float);self.u=np.array(u,float)
    self.nscale=float(la.norm(J,2)); self.M=J/self.nscale
    self.K=self.M.T@self.M
    _,self.ca,self.Pa=pauli_decomp(self.M)
    _,self.cd,self.Pd=pauli_decomp(self.K)
    self.Ga=float(np.abs(self.ca).sum());self.Gd=float(np.abs(self.cd).sum())
    self.training_calls=0;self.train_shots=0
 def sample_exp(self,theta,kind,n):
    v=prep(theta)
    if kind=='a':
        c=self.ca;P=self.Pa
        overlap=np.einsum('i,tij,j->t',self.u,P,v)
        means=np.real(c/np.abs(c)*overlap)
        gamma=self.Ga
    else:
        c=self.cd;P=self.Pd
        means=np.real(c/np.abs(c)*np.einsum('i,tij,j->t',v,P,v))
        gamma=self.Gd
    if np.max(np.abs(means))>1+1e-9:raise RuntimeError('invalid Hadamard/Pauli means')
    p=np.abs(c)/gamma;cnt=self.rng.multinomial(n,p)
    pos=self.rng.binomial(cnt,np.clip((means+1)/2,0,1))
    return float(gamma*(2*np.sum(pos)-n)/n)
 def cost(self,theta,shots):
    a=self.sample_exp(theta,'a',shots);D=self.sample_exp(theta,'D',shots)
    self.training_calls+=1;self.train_shots+=2*shots
    # clip sampled D to avoid numerical singularity; not used in certification.
    return float(np.clip(1-a*a/max(D,.08),-1.5,2)),a,D
 def ideal_cost(self,theta):
    # reporting only; never used in SPSA updates or selection
    v=prep(theta);p=self.M@v; aa=self.u@p;D=p@p
    return float(1-aa*aa/D)

def train(oracle,theta0,iters=250,shots_per_observable=10000,seed=77):
    rg=np.random.default_rng(seed)
    t=theta0.copy(); evaluations=[];checkpoints=[]
    # SPSA minimizes noisy projective least-squares loss; no exact objective in gradient.
    for k in range(1,iters+1):
        ck=.20/(k**.101);ak=.055/(k+12)**.602
        delta=rg.choice([-1.,1.],size=len(t))
        cp,*_=oracle.cost(t+ck*delta,shots_per_observable)
        cm,*_=oracle.cost(t-ck*delta,shots_per_observable)
        grad=((cp-cm)/(2*ck))*delta
        t=np.mod(t-ak*np.clip(grad,-6.,6.)+np.pi,2*np.pi)-np.pi
        if k in [1,20,50,100,150,200,iters]:
            # separate noisy holdout checkpoint; do not use ideal error to select
            loss,a,D=oracle.cost(t,5*shots_per_observable)
            checkpoints.append({'iteration':k,'sampled_loss':float(loss),'sampled_a':float(a),'sampled_D':float(D)})
    return t,checkpoints

def sample_full_residual(J,v,shots_per_coordinate,rng, alpha=.05):
    # Direct 5-system-qubit Hadamard oracle. importance sample full 200 Pauli terms.
    # Use analytic probability in ideal circuit simulator, draw actual finite-shot outcomes.
    from prior.run_experiment import decompose
    Jp=np.zeros((16,16));Jp[:14,:14]=J
    H=np.block([[np.zeros((16,16)),Jp],[Jp.T,np.zeros((16,16))]])
    terms=decompose(H)
    vals=np.array([c for _,c,P in terms]);Gamma=float(abs(vals).sum()); p=abs(vals)/Gamma
    psi=np.r_[np.zeros(16),v].astype(complex)
    mu=np.stack([np.real((P@psi)[:14]) for _,c,P in terms])
    w=np.empty(14)
    for a in range(14):
       counts=rng.multinomial(shots_per_coordinate,p)
       pos=rng.binomial(counts,np.clip((1+np.sign(vals)*mu[:,a])/2,0,1))
       w[a]=Gamma*(2*pos.sum()-shots_per_coordinate)/shots_per_coordinate
    e=Gamma*np.sqrt(2*np.log(2*14/alpha)/shots_per_coordinate)
    return w,float(e),len(terms),Gamma

def measure_amplitudes(v,shots_per_amplitude,rng,alpha=.05):
    # Hadamard test for Re<e_a|v> on 4-qubit state; one ancilla.
    n=14
    pos=rng.binomial(shots_per_amplitude,np.clip((1+v[:n])/2,0,1))
    vh=2*pos/shots_per_amplitude-1
    e=np.sqrt(2*np.log(2*n/alpha)/shots_per_amplitude)
    return vh,float(e)

def holdout(d,oracle,theta,n_a=800000,n_D=800000,n_w=120000,n_v=120000,seed=812,alpha=.05):
    rng=np.random.default_rng(seed)
    v=prep(theta)
    # scale from independent a/D holdout.
    a=oracle.sample_exp(theta,'a',n_a);D=oracle.sample_exp(theta,'D',n_D)
    ea=oracle.Ga*np.sqrt(2*np.log(2/(alpha/4))/n_a)
    eD=oracle.Gd*np.sqrt(2*np.log(2/(alpha/4))/n_D)
    b=d['b'];bn=la.norm(b)
    # Measurable least squares scale s=(bn/||J||) a/D, only when D positive in high confidence
    s=float(bn/oracle.nscale*a/D) if D>0 else float('nan')
    s_interval=[float(bn/oracle.nscale*(a-ea)/(D+eD)),float(bn/oracle.nscale*(a+ea)/(D-eD))] if D>eD else None
    wh,ew,terms,Gamma=sample_full_residual(d['A'],v,n_w,rng,alpha/4)
    vh,ev=measure_amplitudes(v,n_v,rng,alpha/4)
    if np.isfinite(s):
      zhat=s*vh; xcandidate=d['xinit']+zhat
      # Bounds true linear residual of reconstructed and USED classical update; data-dependent scale fine.
      rh=s*wh-b
      component_upper=np.abs(rh)+abs(s)*(ew+np.sum(np.abs(d['A']),axis=1)*ev)
      res_upper_inf=float(np.max(component_upper))
      # Different upper if we directly check computed F -- classically accessible, no quantum measurement necessary.
      actual_linear=float(la.norm(d['A']@zhat-b,np.inf))
      actual_nonlinear=float(la.norm(d['F'](xcandidate),np.inf))
      x_error_inf=float(la.norm(xcandidate-d['sol'],np.inf))
      safe_voltage=bool(np.all((d['unpack'](xcandidate)[1] >=.9)&(d['unpack'](xcandidate)[1] <=1.1)))
      # Conservative necessary threshold for nonlinear enclosure from prior small-uncertainty studies
      target_box=1e-4
      B=la.inv(d['J'](xcandidate))
      # Certified via linear-algebra inequality would need ||B||∞ residual<box, even before nonlinear C.
      necessary_surrogate=float(la.norm(B,np.inf)*res_upper_inf)
      full_box_pass=bool(necessary_surrogate < target_box)
    else:
      zhat=np.full(14,np.nan);rh=np.full(14,np.nan);res_upper_inf=float('inf');actual_linear=float('inf');actual_nonlinear=float('inf');x_error_inf=float('inf');necessary_surrogate=float('inf');full_box_pass=False;safe_voltage=False
    truew=d['A']@v[:14]
    result={
      'training_shots':int(oracle.train_shots),'training_objective_evaluations':int(oracle.training_calls),
      'holdout_shots':int(n_a+n_D+14*n_w+14*n_v),'total_allocation_shots':int(oracle.train_shots+n_a+n_D+14*n_w+14*n_v),
      'holdout_a_estimate':a,'holdout_D_estimate':D,'a_confidence_radius':float(ea),'D_confidence_radius':float(eD),
      'scale_estimate':s,'scale_confidence_interval':s_interval,'scale_interval_valid':s_interval is not None,
      'residual_coordinate_width':ew,'amplitude_coordinate_width':ev,
      'residual_pauli_terms':terms,'residual_pauli_l1_norm':Gamma,
      'max_abs_measured_residual':float(np.max(np.abs(rh))),
      'max_certified_linear_residual_upper':res_upper_inf,
      'actual_linear_residual_from_reconstructed_correction':actual_linear,
      'actual_nonlinear_mismatch_from_reconstructed_correction':actual_nonlinear,
      'actual_state_error_inf_vs_reference':x_error_inf,
      'conservative_norm_error_upper':necessary_surrogate,
      'conservative_sufficient_small_box_test_pass':full_box_pass,
      'tested_box_halfwidth':1e-4,'fully_validated_nonlinear_certificate':False,
      'reconstructed_delta':zhat.tolist(),'xcandidate':xcandidate.tolist(),
      'voltage_point_estimate_within_0p9_1p1':safe_voltage,
      'max_true_scale_residual_raw':float(np.max(np.abs(s*truew-b))) if np.isfinite(s) else None,
      'joint_mean_coverage_observed':bool((abs(a-(oracle.u@(oracle.M@v)))<=ea) and (abs(D-la.norm(oracle.M@v)**2)<=eD) and np.all(np.abs(wh-truew)<=ew) and np.all(np.abs(vh-v[:14])<=ev)),
      'uses_exact_solution_during_training':False,
      'quantum_backend_run':False,
      'quantum_measurements':'Ideal circuit probability samples with independent holdout; no hardware gate/readout noise',
      'full_grid_caveat':'Newton correction is reconstructed by amplitude Hadamard tests; strong certification not achieved.'
    }
    return result

def main():
    d=build_full(); Jp=np.zeros((16,16));Jp[:14,:14]=d['A'];bp=np.r_[d['b'],np.zeros(2)];u=bp/la.norm(bp)
    theta0=angles_from_known_rhs(u)
    assert np.allclose(prep(theta0),u)
    oracle=SampledOracle(Jp,u)
    start_loss=oracle.ideal_cost(theta0)
    t0=time.perf_counter()
    theta,checkpoints=train(oracle,theta0)
    ideal_training_seconds=time.perf_counter()-t0
    hold=holdout(d,oracle,theta)
    # Post-training diagnostic only; no exact target used for optimization or selection
    v=prep(theta)
    true_exact=d['exact']
    rel_target_state=float(abs(np.dot(v[:14],true_exact/la.norm(true_exact))))
    report={'benchmark':'Official MATPOWER case9 model (reduced AC Newton at disturbed state)',
            'full_real_newton_dim':14,'padded_dim':16,'system_qubits_including_fixed_flag':5,
            'ancilla_qubits_for_measurements':1,'variational_state_parameters':15,
            'training_steps':250,'training_shots_per_observable':10000,
            'train_initialized_using_rhs_not_exact_solution':True,
            'loss_initial_diagnostic':start_loss,
            'loss_final_diagnostic':oracle.ideal_cost(theta),
            'overlap_absolute_with_exact_direction_posthoc':rel_target_state,
            'ideal_distribution_simulator_seconds':ideal_training_seconds,
            'sampling_primitive':{'a':int(len(oracle.ca)),'D':int(len(oracle.cd)),
                                  'coefficient_l1_a':oracle.Ga,'coefficient_l1_D':oracle.Gd},
            'checkpoints':checkpoints,'theta_final':theta.tolist(),
            'holdout':hold}
    (HERE/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['theta_final','checkpoints']},indent=2))

if __name__=='__main__':main()
