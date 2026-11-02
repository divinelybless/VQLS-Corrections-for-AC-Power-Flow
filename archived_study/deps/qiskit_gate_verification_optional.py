"""OPTIONAL QISKIT AER: run genuine gate/circuit-based Hadamard tests.
Not executed in the study environment: Qiskit/Qiskit-Aer unavailable.
Install qiskit and qiskit-aer in a network-enabled Python environment.
This is a reference circuit, not a scalable production algorithm.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RYGate, PauliGate, XGate
from qiskit_aer import AerSimulator
from run_multiseed import Oracle, train, get_case, prep

def tree_prep_gate(theta):
    n=int(round(np.log2(len(theta)+1)))
    assert len(theta)==2**n-1
    circ=QuantumCircuit(n,name='VQLS-tree')
    start=0
    for depth in range(n):
        target=n-depth-1
        for prefix in range(2**depth):
            angle=2*theta[start+prefix]
            if depth==0:
                circ.ry(angle,target)
            else:
                ctrl=[n-1-h for h in range(depth)]
                bits=[(prefix>>(depth-1-h))&1 for h in range(depth)]
                ctrl_state=sum(b<<h for h,b in enumerate(bits))
                circ.append(RYGate(angle).control(depth,ctrl_state=ctrl_state),ctrl+[target])
        start+=2**depth
    return circ.to_gate(label='VQLS-prep')

def hadamard_pauli_overlap(theta,pauli_word,coefficient,coordinate):
    n=len(pauli_word); anc=n
    qc=QuantumCircuit(n+1,1)
    qc.h(anc)
    for qubit in range(n):
        if (coordinate>>qubit)&1:
            qc.append(XGate().control(1,ctrl_state=0),[anc,qubit])
    qc.append(tree_prep_gate(theta).control(1),[anc]+list(range(n)))
    qc.append(PauliGate(pauli_word).control(1),[anc]+list(range(n)))
    qc.p(float(np.angle(coefficient)),anc)
    qc.h(anc)
    qc.measure(anc,0)
    return qc

def main():
    sim=AerSimulator(method='statevector')
    for name in ['IEEE9','IEEE14']:
        case=get_case(name);o=Oracle(case['A'],case['b']);theta=train(o,711)
        v=prep(theta)
        terms=[0,len(o.ca)//2, len(o.ca)-1]
        # include at least one imaginary coefficient to test the nonsymmetric Jacobian
        imaginary=np.flatnonzero(abs(o.ca.imag)>1e-10)
        if len(imaginary):terms[1]=int(imaginary[0])
        for t in terms:
            coord=0;lab=o.ca_words[t];coeff=o.ca[t]
            qc=hadamard_pauli_overlap(theta,lab,coeff,coord)
            counts=sim.run(transpile(qc,sim),shots=4000).result().get_counts()
            estimate=(counts.get('0',0)-counts.get('1',0))/4000
            expected=np.real((coeff/abs(coeff))*np.vdot(np.eye(o.p)[coord],o.Pa[t]@v))
            print(name,lab,coeff, 'sampled',round(estimate,5),'expected',round(expected,5),
                  'qubits',qc.num_qubits,'depth',qc.depth())
if __name__=='__main__':main()
