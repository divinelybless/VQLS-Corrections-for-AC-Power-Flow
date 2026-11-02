"""Exploratory classical certification comparison under correlated primitive branch intervals.
Uses mpmath.iv (45 dps) and benchmark-specific data copied from verify_power.
All floating point input data are converted to interval points; does not audit binary input uncertainty.
"""
import numpy as np, scipy.linalg as la, json, time
from scipy.optimize import root
from mpmath import iv
from prior.verify_power import cases, model
iv.dps=45

def point(x): return iv.mpf(float(x))
def ends(x): return float(x.a),float(x.b)
def absub(x):return max(abs(ends(x)[0]),abs(ends(x)[1]))
def isum(xs):
    s=point(0)
    for a in xs:s+=a
    return s

def intersect(a,b):
    lo=max(ends(a)[0],ends(b)[0]); hi=min(ends(a)[1],ends(b)[1]);
    return None if lo>hi else iv.mpf([lo,hi])

def run_case(name,bus,edges,vmset,pg,rel=1e-6,inject=1e-8,radius=1e-4, center=None):
    Y,ang,pq,unpack,F,J,x0=model(bus,edges,vmset,pg)
    result=root(F,x0,jac=J,tol=1e-12)
    if la.norm(F(result.x),np.inf)>1e-9:return {'name':name,'status':'nominal root failure'}
    xhat=result.x if center is None else np.array(center,dtype=float); d=len(xhat); n=len(bus);B=la.inv(J(xhat))
    # Physical primitive uncertainty: each nonzero branch r, x, b changes by common relative interval,
    # each nonunit transformer tap also has common fractional interval. Entries of Y are correlated.
    def prim(x, nonzero=True):
        a=float(x); delta=abs(a)*rel if nonzero and a else 0
        return iv.mpf([a-delta,a+delta])
    Yr=[[point(0) for j in range(n)] for i in range(n)]
    Yi=[[point(0) for j in range(n)] for i in range(n)]
    for i in range(n):
        Yr[i][i]+=point(bus[i,3]/100)
        Yi[i][i]+=point(bus[i,4]/100)
    for f,t,r,x,b,tap in edges:
        i=int(f)-1;j=int(t)-1
        rr=prim(r); xx=prim(x); bb=prim(b)
        tt=prim(tap) if tap else point(1)
        denominator=rr*rr+xx*xx
        yg=rr/denominator; yb=-xx/denominator
        # shunt charging b/2 and tap scaling
        aa=tt*tt
        Yr[i][i]+=yg/aa; Yi[i][i]+=(yb+bb/2)/aa
        Yr[j][j]+=yg; Yi[j][j]+=yb+bb/2
        Yr[i][j]-=yg/tt; Yi[i][j]-=yb/tt
        Yr[j][i]-=yg/tt; Yi[j][i]-=yb/tt
    def expand(x):
        th=[point(z) for z in np.deg2rad(bus[:,6])]
        v=[point(z) for z in bus[:,5]]
        for k,z in vmset.items(): v[k-1]=point(z)
        for k,i in enumerate(ang):th[i]=x[k]
        for k,i in enumerate(pq):v[i]=x[len(ang)+k]
        return th,v
    def fj(x,derivative=False):
        th,v=expand(x);P=[];Q=[]
        for i in range(n):
            pi=point(0);qi=point(0)
            for j in range(n):
                dt=th[i]-th[j]; co=iv.cos(dt);si=iv.sin(dt)
                pi+=v[i]*v[j]*(Yr[i][j]*co+Yi[i][j]*si)
                qi+=v[i]*v[j]*(Yr[i][j]*si-Yi[i][j]*co)
            P.append(pi);Q.append(qi)
        p=(np.array([pg.get(i+1,0) for i in range(n)])-bus[:,1])/100
        q=-bus[:,2]/100
        ff=[P[i]-iv.mpf([float(p[i]-inject),float(p[i]+inject)]) for i in ang]
        ff += [Q[i]-iv.mpf([float(q[i]-inject),float(q[i]+inject)]) for i in pq]
        if not derivative:return ff
        JJ=[]
        for ty,i in [('P',i) for i in ang]+[('Q',i) for i in pq]:
            row=[]
            for mode,k in [('theta',k) for k in ang]+[('v',k) for k in pq]:
                if mode=='theta':
                    if i!=k:
                        dt=th[i]-th[k]
                        z=v[i]*v[k]*(Yr[i][k]*iv.sin(dt)-Yi[i][k]*iv.cos(dt)) if ty=='P' else -v[i]*v[k]*(Yr[i][k]*iv.cos(dt)+Yi[i][k]*iv.sin(dt))
                    else:
                        z=point(0)
                        for j in range(n):
                            if j==i:continue
                            dt=th[i]-th[j]
                            z+=(-v[i]*v[j]*(Yr[i][j]*iv.sin(dt)-Yi[i][j]*iv.cos(dt)) if ty=='P' else v[i]*v[j]*(Yr[i][j]*iv.cos(dt)+Yi[i][j]*iv.sin(dt)))
                else:
                    if i!=k:
                        dt=th[i]-th[k]
                        z=v[i]*(Yr[i][k]*iv.cos(dt)+Yi[i][k]*iv.sin(dt)) if ty=='P' else v[i]*(Yr[i][k]*iv.sin(dt)-Yi[i][k]*iv.cos(dt))
                    else:
                        z=point(2)*v[i]*(Yr[i][i] if ty=='P' else -Yi[i][i])
                        for j in range(n):
                            if j==i:continue
                            dt=th[i]-th[j]
                            z+=v[j]*(Yr[i][j]*iv.cos(dt)+Yi[i][j]*iv.sin(dt)) if ty=='P' else v[j]*(Yr[i][j]*iv.sin(dt)-Yi[i][j]*iv.cos(dt))
                row.append(z)
            JJ.append(row)
        return ff,JJ
    tic=time.perf_counter()
    ff=fj([point(z) for z in xhat]);bx=[iv.mpf([float(z-radius),float(z+radius)]) for z in xhat]
    _,jj=fj(bx,True)
    M=[[isum(point(B[i,k])*jj[k][j] for k in range(d)) for j in range(d)] for i in range(d)]
    s=[-isum(point(B[i,k])*ff[k] for k in range(d)) for i in range(d)]
    A=[[point(i==j)-M[i][j] for j in range(d)] for i in range(d)]
    a=np.array([[absub(z) for z in row] for row in A]);v=np.array([absub(z) for z in s]);
    kraw=[s[i]+isum(A[i][j]*iv.mpf([-radius,radius]) for j in range(d)) for i in range(d)]
    kinside=all(ends(z)[0]>-radius and ends(z)[1]<radius for z in kraw)
    # Hansen-Sengupta: solve interval preconditioned Newton linear inclusion M y = s
    # using interval Gauss-Seidel contractor, iteratively intersects with initial displacement box.
    D=[iv.mpf([-radius,radius]) for i in range(d)]
    valid=True;iters=0
    for it in range(15):
        change=0
        for i in range(d):
            rhs=s[i]-isum(M[i][j]*D[j] for j in range(d) if j!=i)
            candidate=rhs/M[i][i] if (ends(M[i][i])[0]>0 or ends(M[i][i])[1]<0) else None
            new=intersect(D[i],candidate) if candidate is not None else D[i]
            if new is None:valid=False;break
            change=max(change,max(abs(ends(D[i])[k]-ends(new)[k]) for k in (0,1)))
            D[i]=new
        iters=it+1
        if not valid or change<1e-17:break
    # Interval Newton enclosure (full interval inverse not computed); HS is interval Gauss-Seidel
    # applied to preconditioned interval Newton system. No independent full inverse comparison claimed.
    widths=[(ends(z)[1]-ends(z)[0])/2 for z in D]; k_widths=[(ends(z)[1]-ends(z)[0])/2 for z in kraw]
    selected=d-1
    comp=la.solve(np.eye(d)-a,v) if max(abs(la.eigvals(a)))<1 else None
    return {'name':name,'relative_primitive_uncertainty':rel,'injection_pu':inject,'state_dimension':d,'krawczyk_strict_inclusion':kinside,
       'krawczyk_inclusion_margin_min':float(min(min(radius+ends(z)[0],radius-ends(z)[1]) for z in kraw)), 'krawczyk_selected_halfwidth':k_widths[selected], 'hansen_sengupta_selected_halfwidth':widths[selected],
       'hansen_sengupta_contracts':valid, 'hansen_sengupta_iterations':iters,
       'componentwise_selected_halfwidth':float(comp[selected]) if comp is not None else None,
       'global_uniform_halfwidth':float(np.max(v/(1-a.sum(axis=1)))) if max(a.sum(axis=1))<1 else None,
       'interval_majorant_rowsum':float(max(a.sum(axis=1))), 'wall_seconds':round(time.perf_counter()-tic,3),
       'assumptions':'Independent relative intervals on primitive branch r/x/charging/taps; structural Y dependencies evaluated by interval arithmetic; nominal PV/PQ types fixed.',
       'scope':'High-precision interval enclosure, not independently audited IEEE standard validated input data.'}
if __name__=='__main__':
    out=[]
    for name,(b,e,vm,p) in cases().items():
        for level in [1e-7,1e-6,1e-5]:
            print('running',name,level,flush=True)
            v=run_case(name,b,e,vm,p,rel=level,inject=level*.01)
            print(v,flush=True);out.append(v)
    with open('/mnt/data/physical_classical_results.json','w') as f:json.dump(out,f,indent=2)
