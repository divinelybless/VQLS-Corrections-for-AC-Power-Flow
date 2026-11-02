"""Explicit gate-by-gate NumPy simulation of ancilla-controlled tree ansatz and Pauli-Hadamard overlap.
This executes gate state updates (including all multiplexed RY and controlled Pauli gates),
not Qiskit or physical quantum hardware. It tests the *complex* Pauli phase convention.
"""
import json, numpy as np
from pathlib import Path
from run_multiseed import Oracle,train,prep,get_case,BASE
from prior.run_experiment import PAULI

H=np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
def anc_h(s):return H@s

def conditional_x(vec,k,n):
    bits=np.arange(2**n)
    vec[:]=vec[bits^(1<<(n-1-k))]

def apply_multiplexed_ry(branch,theta,n):
    i=0
    for depth in range(n):
        mask=(1<<(n-1-depth)); nodes=2**depth
        for prefix in range(nodes):
            t=theta[i+prefix];c=np.cos(t);sn=np.sin(t)
            for low in range(2**(n-depth-1)):
                a=(prefix<<(n-depth))|low
                b=a|mask
                u,v=branch[a],branch[b]
                branch[a]=c*u-sn*v
                branch[b]=sn*u+c*v
        i+=nodes

def apply_pauli_word(branch,word):
    n=len(word)
    for k,w in enumerate(word):
        bit=1<<(n-k-1)
        if w=='I':continue
        b=np.arange(2**n)
        if w=='Z':branch[:]=branch*((1-2*((b&bit)!=0)))
        if w=='X':branch[:]=branch[b^bit]
        if w=='Y':branch[:]=(1j*(2*((b&bit)!=0)-1))*branch[b^bit]

def overlap_gate(theta, word, idx, phase=0.0):
    n=len(word);data=2**n
    states=np.zeros((2,data),complex);states[0,0]=1
    states=anc_h(states)
    # Controlled preparation on ancilla branch |0> : |e_idx>
    for k in range(n):
        if (idx>>(n-k-1))&1:conditional_x(states[0],k,n)
    # Controlled real Ry tree prep on branch |1>: U(theta)|0>
    apply_multiplexed_ry(states[1],theta,n)
    apply_pauli_word(states[1],word)
    states[1]*=np.exp(1j*phase)
    states=anc_h(states)
    p0=float(np.vdot(states[0],states[0]).real)
    assert abs(np.linalg.norm(states)-1)<1e-10
    return 2*p0-1

def main():
    out={}
    for name in ['IEEE9','IEEE14']:
        case=get_case(name);o=Oracle(case['A'],case['b']);th=train(o,711)
        v=prep(th);rng=np.random.default_rng(31415)
        indices=[0,len(case['b'])//2,len(case['b'])-1]
        # Evaluate 3 Pauli terms including purely imaginary coefficient terms
        selected=[int(np.argmax(abs(o.ca))),int(np.flatnonzero(abs(o.ca.imag)>1e-12)[0]),len(o.ca)//2]
        tests=[]
        for a in indices:
            for t in selected:
                word=o.ca_words[t];coeff=o.ca[t];phase=np.angle(coeff)
                observed=overlap_gate(th,word,a,phase)
                ideal=float(np.real((coeff/abs(coeff))*(o.Pa[t]@v)[a]))
                assert np.isclose(observed,ideal,atol=1e-12),(observed,ideal,word)
                N=2500
                shot_avg=(2*rng.binomial(N,(1+observed)/2)-N)/N
                tests.append(dict(index=a,pauli_word=word,coefficient_real=float(coeff.real),coefficient_imag=float(coeff.imag),
                         exact_expected_value=ideal,explicit_gate_expected_value=float(observed),
                         sampled_estimate=float(shot_avg),shots=N))
        out[name]={'data_qubits':len(o.ca_words[0]), 'ancilla_qubits':1,'number_explicit_gate_circuits':len(tests),
                'maximum_probability_discrepancy':float(max(abs(t['explicit_gate_expected_value']-t['exact_expected_value']) for t in tests)),
                'tests':tests}
        print(name,len(tests),'max gate/analytic discrepancy',out[name]['maximum_probability_discrepancy'],flush=True)
    (BASE/'gate_level_results.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
