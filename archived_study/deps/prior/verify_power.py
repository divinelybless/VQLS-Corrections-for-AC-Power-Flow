import numpy as np, scipy.linalg as la
from scipy.optimize import root
from pathlib import Path
import json
ROOT=Path(__file__).parent
# Official MATPOWER case9 and case14 source parameters. Bus tuples: type, Pd, Qd, Gs, Bs, Vm, Va.
def cases():
 b9=np.array([[3,0,0,0,0,1,0],[2,0,0,0,0,1,0],[2,0,0,0,0,1,0],[1,0,0,0,0,1,0],[1,90,30,0,0,1,0],[1,0,0,0,0,1,0],[1,100,35,0,0,1,0],[1,0,0,0,0,1,0],[1,125,50,0,0,1,0]],float)
 e9=np.array([[1,4,0,.0576,0,0],[4,5,.017,.092,.158,0],[5,6,.039,.17,.358,0],[3,6,0,.0586,0,0],[6,7,.0119,.1008,.209,0],[7,8,.0085,.072,.149,0],[8,2,0,.0625,0,0],[8,9,.032,.161,.306,0],[9,4,.01,.085,.176,0]],float)
 b14=np.array([[3,0,0,0,0,1.06,0],[2,21.7,12.7,0,0,1.045,-4.98],[2,94.2,19,0,0,1.01,-12.72],[1,47.8,-3.9,0,0,1.019,-10.33],[1,7.6,1.6,0,0,1.02,-8.78],[2,11.2,7.5,0,0,1.07,-14.22],[1,0,0,0,0,1.062,-13.37],[2,0,0,0,0,1.09,-13.36],[1,29.5,16.6,0,19,1.056,-14.94],[1,9,5.8,0,0,1.051,-15.1],[1,3.5,1.8,0,0,1.057,-14.79],[1,6.1,1.6,0,0,1.055,-15.07],[1,13.5,5.8,0,0,1.05,-15.16],[1,14.9,5,0,0,1.036,-16.04]],float)
 e14=np.array([[1,2,.01938,.05917,.0528,0],[1,5,.05403,.22304,.0492,0],[2,3,.04699,.19797,.0438,0],[2,4,.05811,.17632,.034,0],[2,5,.05695,.17388,.0346,0],[3,4,.06701,.17103,.0128,0],[4,5,.01335,.04211,0,0],[4,7,0,.20912,0,.978],[4,9,0,.55618,0,.969],[5,6,0,.25202,0,.932],[6,11,.09498,.1989,0,0],[6,12,.12291,.25581,0,0],[6,13,.06615,.13027,0,0],[7,8,0,.17615,0,0],[7,9,0,.11001,0,0],[9,10,.03181,.0845,0,0],[9,14,.12711,.27038,0,0],[10,11,.08205,.19207,0,0],[12,13,.22092,.19988,0,0],[13,14,.17093,.34802,0,0]],float)
 return {'IEEE9':(b9,e9,{1:1.04,2:1.025,3:1.025},{1:72.3,2:163,3:85}), 'IEEE14':(b14,e14,{1:1.06,2:1.045,3:1.01,6:1.07,8:1.09},{1:232.4,2:40,3:0,6:0,8:0})}

def buildY(bus,edge):
 n=len(bus);Y=np.diag((bus[:,3]+1j*bus[:,4])/100).astype(complex)
 for f,t,r,x,b,tap in edge:
  i=int(f)-1;j=int(t)-1;a=tap if tap else 1.;y=1/(r+1j*x);ch=1j*b/2
  Y[i,i]+=(y+ch)/(a*a);Y[j,j]+=y+ch;Y[i,j]-=y/a;Y[j,i]-=y/a
 return Y

def model(bus,edge,vmset,pg):
 Y=buildY(bus,edge);n=len(bus);types=bus[:,0];ang=np.flatnonzero(types!=3);pq=np.flatnonzero(types==1)
 vm=bus[:,5].copy();theta=np.deg2rad(bus[:,6].copy());
 for k,v in vmset.items():vm[k-1]=v
 p=(np.array([pg.get(i+1,0) for i in range(n)])-bus[:,1])/100
 q=-bus[:,2]/100
 def unpack(x):
  th=theta.copy();v=vm.copy();th[ang]=x[:len(ang)];v[pq]=x[len(ang):];return th,v
 def fun(x):
  th,v=unpack(x);vv=v*np.exp(1j*th);s=vv*np.conj(Y@vv);return np.r_[s.real[ang]-p[ang],s.imag[pq]-q[pq]]
 def jac(x):
  th,v=unpack(x);vv=v*np.exp(1j*th);I=Y@vv;S=vv*np.conj(I)
  # complex derivative arrays dS/dtheta, dS/dv
  dtheta=1j*np.diag(vv)@(np.diag(np.conj(I))-np.conj(Y@np.diag(vv)))
  dv=np.diag(vv/v)@np.diag(np.conj(I))+np.diag(vv)@np.conj(Y@np.diag(np.exp(1j*th)))
  return np.block([[dtheta.real[np.ix_(ang,ang)],dv.real[np.ix_(ang,pq)]],[dtheta.imag[np.ix_(pq,ang)],dv.imag[np.ix_(pq,pq)]]])
 x0=np.r_[theta[ang],vm[pq]]
 return Y,ang,pq,unpack,fun,jac,x0

def evaluate(name,bus,edge,vmset,pg):
 Y,ang,pq,unpack,F,J,x0=model(bus,edge,vmset,pg)
 sol=root(F,x0,jac=J,tol=1e-12);assert la.norm(F(sol.x),np.inf)<1e-9,(name,sol.message,F(sol.x))
 x=sol.x; d=len(x);JJ=J(x);B=la.inv(JJ);cond=np.linalg.cond(JJ)
 # Synthetic, clearly stated uncertainty: each power mismatch <=1e-8 p.u., measurement <=1e-9 p.u.
 # Conservative derivative majorant: each Hessian entry <= H=16*n^2*max |Y_ij| * vmax^2
 # from product and trigonometric differentiation on v<=1.2; intentionally very conservative.
 n=len(bus);vmax=1.2;H=16*n*n*np.max(np.abs(Y))*vmax*vmax
 # Each |J_ab(y)-J_ab(x)| <= H*||y-x||_inf ; global C = |I-BJ| + |B| @ (H * d * max t matrix) .
 eps_model=1e-8;eps_shot=1e-9
 residual=np.full(d,eps_model+eps_shot)
 t0=np.abs(B)@residual
 t=np.maximum(2*t0,1e-11*np.ones(d))
 for iteration in range(30):
  C=np.abs(np.eye(d)-B@JJ)+np.abs(B)@(np.full((d,d),H*np.max(t)))
  target=np.abs(B)@residual + C@t
  if np.all(target<=t):break
  t=np.maximum(t*1.5, target*1.1)
 valid=np.all(np.abs(B)@residual+C@t<=t) and np.linalg.norm(C,np.inf)<1
 # Localized topology sensitivity, with C majorant, h = g(I-C)^-1 |B|
 Q=la.solve(np.eye(d)-C,np.abs(B)); h=np.sum(Q[len(ang):,:],axis=0) if len(pq) else Q[0,:]
 # For voltage coordinate at PQ bus, g=e_k rather than sum: choose largest sensitivity voltage index
 chosen=len(ang)+int(np.argmax(np.sum(Q[len(ang):,:],axis=1)))
 h=Q[chosen,:]
 a=h.copy();active=a>1e-14
 Nopt=(np.sum(a[active]**(2/3))**2)*(a[active]**(2/3)) # tau=1, shot coefficients identical
 Nuniform=d*(np.sum(a[active]))**2 # uniform across all d coords
 ratio=float(np.sum(Nopt)/Nuniform)
 # Euclidean enclosure <= ||t||_2, voltage-specific t_k is sharper
 ratio_volt=float(t[chosen]/la.norm(t))
 return dict(name=name,buses=n,branches=len(edge),dimension=d,Y_nnz=int(np.count_nonzero(np.abs(Y)>1e-12)),J_nnz=int(np.count_nonzero(np.abs(JJ)>1e-12)),residual_inf=float(la.norm(F(x),np.inf)),condition=float(cond),radius_max=float(max(t)),voltage_radius=float(t[chosen]),euclidean_radius=float(la.norm(t)),max_C_rowsum=float(np.max(np.sum(C,axis=1))),valid=bool(valid),shot_ratio=ratio,shot_saving_pct=(1-ratio)*100,voltage_box_to_euclid=ratio_volt,selected_pq_bus=int(pq[chosen-len(ang)]+1),v_min=float(min(unpack(x)[1])),v_max=float(max(unpack(x)[1])),max_hess_majorant=float(H))

if __name__=='__main__':
 data=[]
 for name,(b,e,v,p) in cases().items():
  o=evaluate(name,b,e,v,p);data.append(o);print(json.dumps(o,indent=2))
 (ROOT/'verification_results.json').write_text(json.dumps(data,indent=2))
