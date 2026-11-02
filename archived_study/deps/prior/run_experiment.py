"""Full IEEE9 Newton-system residual Hadamard oracle, using NumPy circuit-probability sampling.
No Qiskit dependency. NO quantum advantage or full stochastic QVLS training claimed.
The trial amplitude is classically prepared for oracle CALIBRATION ONLY.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import root
from scipy.linalg import solve
HERE=Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_power import cases, model

II=np.eye(2, dtype=complex)
X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]],complex)
Z=np.diag([1,-1]).astype(complex)
PAULI={'I':II,'X':X,'Y':Y,'Z':Z}

def pauli_matrix(word):
    P=np.array([[1.]],complex)
    for c in word: P=np.kron(P,PAULI[c])
    return P

def decompose(H,cutoff=1e-10):
    from itertools import product
    d=H.shape[0]; n=int(np.log2(d)); terms=[]
    for tup in product('IXYZ',repeat=n):
        lab=''.join(tup); P=pauli_matrix(lab)
        c=np.trace(P@H)/d
        if abs(c)>cutoff:
            if abs(c.imag)>1e-9: raise RuntimeError('non-Hermitian decomposition')
            terms.append((lab, float(c.real),P))
    assert np.allclose(sum(c*P for _,c,P in terms),H,atol=1e-10)
    return terms

def build_full():
    bus,edge,vm,pg=cases()['IEEE9']
    _,ang,pq,unpack,F,J,x0=model(bus,edge,vm,pg)
    sol=root(F,x0,jac=J,tol=1e-12)
    assert np.max(np.abs(F(sol.x)))<1e-10
    n=len(sol.x)
    perturb=1.2e-3*np.sin(np.arange(n)*0.73+0.5)
    xinit=sol.x + perturb
    b=-F(xinit); A=J(xinit)
    exact=solve(A,b)
    assert n==14
    Jpad=np.zeros((16,16));Jpad[:n,:n]=A
    H=np.block([[np.zeros((16,16)),Jpad],[Jpad.T,np.zeros((16,16))]])
    f=np.r_[b,np.zeros(18)]
    return dict(H=H,f=f,A=A,b=b,xinit=xinit,sol=sol.x,F=F,J=J,unpack=unpack,exact=exact,ang=ang,pq=pq)

def candidate_full(d,calibration_level=.015):
    # Constructed using an exact classical correction SOLELY to validate oracle mathematics.
    # This does NOT constitute a quantum linear solve or training result.
    exact=d['exact']; n=len(exact)
    seed=np.cos(np.arange(n)*1.31+0.37)
    delta=exact+calibration_level*np.linalg.norm(exact)*seed/np.linalg.norm(seed)
    v=np.r_[delta/np.linalg.norm(delta),np.zeros(2)]
    psi=np.r_[np.zeros(16),v].astype(complex)
    return psi,delta

def overlap_distribution(i,lab,c,P,psi,shots,rng):
    # Exact 6-qubit Hadamard-test branch simulation: (|0>|i>+|1>P|psi>)/sqrt(2)
    # Measure ancilla in X basis. The outcome mean is Re<e_i|P|psi>.
    mu=float((P@psi)[i].real)
    if not -1.0000001 <= mu <= 1.0000001: raise ValueError(mu)
    positive=rng.binomial(shots,(1+max(-1,min(1,mu)))/2)
    return c*(2*positive/shots-1),mu

def once(terms,psi,b,shot_per_group,seed,alpha=.05):
    rng=np.random.default_rng(seed);m=len(b)
    w_hat=np.zeros(m)
    for i in range(m):
        for lab,c,P in terms:
            contribution,mu=overlap_distribution(i,lab,c,P,psi,shot_per_group,rng)
            w_hat[i]+=contribution
    bound=np.sqrt(2*np.log(2*m*len(terms)/alpha)/shot_per_group)*sum(abs(c) for _,c,_ in terms)
    Wlo=w_hat-bound; Whi=w_hat+bound
    # Scale from the SAME measurement record (correlated scale/residual is fine by joint event).
    denom=float(w_hat@w_hat)
    shat=float(b@w_hat/denom) if denom>0 else 0.
    # Valid residual for the ACTUALLY APPLIED shat. No extra s_star scale error needed.
    rhat=shat*w_hat-b
    r_eps=abs(shat)*bound
    # Optional optimal-scale interval. Denominator lower estimate has to be positive.
    dlo=float(np.sum(np.where((Wlo<=0)&(Whi>=0),0,np.minimum(Wlo**2,Whi**2))))
    dhi=float(np.sum(np.maximum(Wlo**2,Whi**2)))
    alo=float(np.sum(np.minimum(b*Wlo,b*Whi)))
    ahi=float(np.sum(np.maximum(b*Wlo,b*Whi)))
    if dlo>0:
        corners=[aa/dd for aa in [alo,ahi] for dd in [dlo,dhi]]
        s_interval=[float(min(corners)),float(max(corners))]
    else: s_interval=None
    # Conservative sufficient linearization mismatch computed at same x0:
    #  |F(x0+s v)| <= |shat*w-b| + |shat|e + (L/2)|shat|^2||v||^2
    return dict(w_hat=w_hat,abs_w_confidence=bound,shat=shat,scale_interval=s_interval,
                rhat=rhat, r_eps=r_eps,measurement_shots=int(m*len(terms)*shot_per_group),
                group_shots=int(shot_per_group), alpha=alpha)

def main():
    t0=time.perf_counter();d=build_full();H=d['H'];psi,delta=candidate_full(d)
    terms=decompose(H);n=len(d['b']);w=(H@psi)[:n].real
    assert np.allclose(w,d['A']@psi[16:16+n],atol=1e-12)
    scale_star=float(d['b']@w/(w@w))
    optimal_res=scale_star*w-d['b']
    # Circuit probability sanity check on nonzero term: explicit 6-qubit controlled-branch state.
    examples=[]
    for i in [0,5,13]:
      for lab,c,P in terms:
       mu=float((P@psi)[i].real)
       if abs(mu)>.15:
        anc0=np.eye(32)[i]/np.sqrt(2)
        anc1=P@psi/np.sqrt(2)
        probability_plus=float(np.linalg.norm((anc0+anc1)/np.sqrt(2))**2)
        assert abs((2*probability_plus-1)-mu)<1e-12
        examples.append({'residual_index':i,'term':lab,'exact_overlap':mu,'x_plus_probability':probability_plus})
        break
    rows=[]
    for group_shots in [64,512,4096]:
      measurement=once(terms,psi,d['b'],group_shots,seed=907+group_shots)
      s=measurement['shat'];truew=w;eps=measurement['abs_w_confidence'];r_true=s*truew-d['b'];bound=abs(measurement['rhat'])+abs(s)*eps
      xcandidate=d['xinit']+s*psi[16:30].real
      fullF=d['F'](xcandidate)
      e0=float(np.max(np.abs(xcandidate-d['sol'])))
      # Only a necessary, easy-to-check residual accuracy test, NOT Krawczyk proof.
      accept_proxy=bool(np.max(bound)<1e-3)
      assert np.all(np.abs(r_true-measurement['rhat'])<=measurement['r_eps']+1e-9)
      rows.append({'shots_per_group':group_shots,'total_residual_shots':measurement['measurement_shots'],
        'measured_scale':s,'scale_interval_exists':measurement['scale_interval'] is not None,
        'absolute_scale_interval':measurement['scale_interval'],
        'max_measured_residual':float(np.max(np.abs(measurement['rhat']))),
        'max_true_linear_residual':float(np.max(np.abs(r_true))),
        'max_certified_linear_residual_upper_bound':float(np.max(bound)),
        'max_abs_overlap_group_bound':float(eps),
        'max_actual_nonlinear_residual':float(np.max(np.abs(fullF))),
        'max_actual_state_error':e0,
        'proxy_residual_under_1e_minus3':accept_proxy,
        'formal_grid_certified':False})
    gamma=sum(abs(c) for _,c,_ in terms)
    # Necessary uniform per Pauli coordinate shots to make a single sufficient residual uncertainty eta:
    # |s| gamma sqrt(2 log(2 m L /alpha)/N) <= eta.
    eta=1e-3;alpha=.05
    shots_group_lb=int(np.ceil((abs(scale_star)*gamma/eta)**2 * 2*np.log(2*n*len(terms)/alpha)))
    result={'case':'MATPOWER case9','full_reduced_newton_dim':n,'embedded_dim':32,'system_qubits':5,
      'hadamard_ancilla_qubits':1,'pauli_terms':len(terms),'pauli_l1_norm':gamma,
      'jacobian_condition_2':float(np.linalg.cond(d['A'])),
      'x0_max_mismatch':float(np.max(np.abs(-d['b']))),
      'classical_Newton_correction_norm':float(np.linalg.norm(d['exact'])),
      'calibration_candidate_prepared_from_classical_exact_correction':True,
      'candidate_scale_optimal_exact':scale_star,
      'optimal_candidate_max_linear_residual':float(np.max(np.abs(optimal_res))),
      'optimal_candidate_max_nonlinear_residual':float(np.max(np.abs(d['F'](d['xinit']+scale_star*psi[16:30].real)))),
      'nonzero_overlap_circuit_sanity':examples,
      'shot_runs':rows,
      'uniform_per_group_sufficient_shots_for_residual_measurement_width_1e-3':shots_group_lb,
      'uniform_total_sufficient_shots_for_residual_measurement_width_1e-3':int(shots_group_lb*n*len(terms)),
      'sampling_alpha':alpha,'quantum_circuits_executed_in_qiskit':False,
      'sampling_simulated_from_explicit_Hadamard_test_probabilities':True,
      'uncertainty_model':'no device noise or bias; ideal preparation of calibration state; sampling uncertainty only',
      'certification_status':'conditional probability result; no outward-rounded parameter-uniform Krawczyk included',
      'runtime_seconds':time.perf_counter()-t0}
    (HERE/'oracle_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['shot_runs','nonzero_overlap_circuit_sanity']},indent=2))
    print('runs:',json.dumps(rows,indent=2))
    # Store reusable numerical data (not user identifying).
    np.savez_compressed(HERE/'oracle_matrices.npz',J=d['A'],H=H,b=d['b'],
                        psi=psi, xinit=d['xinit'], exact=d['exact'],
                        pauli_labels=np.array([lab for lab,_,_ in terms]),
                        pauli_coeffs=np.array([c for _,c,_ in terms]))
    return result
if __name__=='__main__':main()
