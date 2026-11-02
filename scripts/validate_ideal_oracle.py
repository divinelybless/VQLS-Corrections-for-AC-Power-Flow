"""Numerical validation of the exact ideal-outcome sampler used in the archive.
This is a sampler check, not gate execution or hardware validation.
"""
import os,sys,json
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
T=Path(__file__).resolve().parent
sys.path.insert(0,str(T/'archived_study/deps'))
from run_multiseed import Oracle
from compare import get_case, prep, angles_from_rhs
result={}
for name in ['IEEE9','IEEE14']:
 d=get_case(name);o=Oracle(d['A'],d['b']);v=prep(angles_from_rhs(o.u))
 rng=np.random.default_rng(49821)
 n=1600;reps=280
 arr_a=np.array([o.sample(v,'a',n,rng) for _ in range(reps)])
 arr_D=np.array([o.sample(v,'D',n,rng) for _ in range(reps)])
 exact_a=float(o.u@(o.M@v)); exact_D=float(v@(o.K@v))
 # For any variable of value +/-Gamma, theoretical Var(Y)=Gamma^2 - mean^2.
 def test(arr,exact,gamma):
  se=((gamma**2-exact**2)/(n*reps))**.5
  z=abs(arr.mean()-exact)/se
  assert z<4.5,(name,exact,arr.mean(),z)
  return dict(true=exact,empirical_mean=float(arr.mean()),standard_error_of_empirical_mean=se,mean_zscore=float(z),empirical_single_batch_variance=float(arr.var(ddof=1)),theoretical_single_batch_variance=(gamma**2-exact**2)/n)
 # Coordinate i=0, a signed interference overlap with exact mean real(v[i]).
 coord=0; amp_means=[]
 for _ in range(reps):
  p=(v[coord]+1)/2
  amp_means.append(2*rng.binomial(n,p)/n-1)
 amp=np.asarray(amp_means)
 result[name]=dict(n=n,repetitions=reps,Pauli_terms_M=len(o.ca),Pauli_terms_K=len(o.cd),imaginary_coeff_M=int(sum(abs(o.ca.imag)>1e-12)),gamma_M=o.Ga,gamma_K=o.Gd,
 numerator=test(arr_a,exact_a,o.Ga),denominator=test(arr_D,exact_D,o.Gd),amplitude_coordinate_0=test(amp,float(v[0]),1.),note='All random draws generated from analytic ideal-outcome probabilities; not measured through gates')
(T/'oracle_sampler_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps({n:{k:v for k,v in o.items() if k in ['Pauli_terms_M','Pauli_terms_K','imaginary_coeff_M','numerator','denominator','amplitude_coordinate_0']} for n,o in result.items()},indent=2))
