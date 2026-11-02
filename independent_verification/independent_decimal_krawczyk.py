"""Independent proof-oriented Krawczyk checker. No mpmath.iv or prior interval code.

All linear ops are Decimal intervals with directed rounding. Trig endpoints are
bounded by rational Taylor polynomials plus an exact Lagrange remainder.
Input parameters are parsed directly from decimal source tokens and candidate
centers / real preconditioners are imported as *exact* binary IEEE754 numbers.
This validates strict root inclusion for the explicitly defined Decimal primitive
parameter box; other claims (e.g., full voltage/branch constraints) need audits.
"""
from decimal import Decimal as D, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as Fr
import ast, math, json, sys, time
from functools import lru_cache
from pathlib import Path
import numpy as np

PREC=65
FLOOR=ROUND_FLOOR; CEIL=ROUND_CEILING

def calc(op, *args, rounding=FLOOR):
    with localcontext() as c:
        c.prec=PREC; c.rounding=rounding
        return op(*args)

def fdiv(fr, rnd):
    return calc(lambda a,b:a/b, D(fr.numerator), D(fr.denominator), rounding=rnd)

class I:
    __slots__=('l','u')
    def __init__(self,l,u=None):
        self.l=l if isinstance(l,D) else D(str(l)); self.u=(self.l if u is None else (u if isinstance(u,D) else D(str(u))))
        assert self.l<=self.u, (self.l,self.u)
    def __add__(a,b):
        b=asI(b);return I(calc(lambda x,y:x+y,a.l,b.l),calc(lambda x,y:x+y,a.u,b.u,rounding=CEIL))
    __radd__=__add__
    def __neg__(a):return I(-a.u,-a.l)
    def __sub__(a,b):return a+-asI(b)
    def __rsub__(a,b):return asI(b)+-a
    def __mul__(a,b):
        b=asI(b); z=[(x,y) for x in (a.l,a.u) for y in (b.l,b.u)]
        return I(min(calc(lambda x,y:x*y,x,y) for x,y in z),max(calc(lambda x,y:x*y,x,y,rounding=CEIL) for x,y in z))
    __rmul__=__mul__
    def __truediv__(a,b):
        b=asI(b)
        if b.l<=0<=b.u: raise ZeroDivisionError((b.l,b.u))
        recip=I(calc(lambda x:D(1)/x,b.u),calc(lambda x:D(1)/x,b.l,rounding=CEIL))
        return a*recip
    def __rtruediv__(a,b):return asI(b)/a
    def __repr__(a):return f'[{a.l},{a.u}]'
    def mag(a):return max(abs(a.l),abs(a.u))
    def width(a):return a.u-a.l

def asI(x):return x if isinstance(x,I) else I(x)
ZERO=I(0); ONE=I(1)

@lru_cache(maxsize=50000)
def point_taylor(x,kind):
    # Exact rational polynomial; Lagrange remainder with a rigorous bound for |x|<=2.
    xx=Fr(x); nterms=20
    assert abs(xx) <= 2, (kind,x)
    result=Fr(0)
    if kind=='sin':
        term=xx
        for k in range(nterms):
            result+=term
            term*= -xx*xx/((2*k+2)*(2*k+3))
        remainder=abs(xx)**(2*nterms)/math.factorial(2*nterms)
    else:
        term=Fr(1)
        for k in range(nterms):
            result+=term
            term*= -xx*xx/((2*k+1)*(2*k+2))
        remainder=abs(xx)**(2*nterms-1)/math.factorial(2*nterms-1)
    return fdiv(result-remainder,FLOOR),fdiv(result+remainder,CEIL)

def sin(a):
    a=asI(a)
    if a.l < D('-1.5') or a.u > D('1.5'):return I(-1,1)
    return I(point_taylor(a.l,'sin')[0],point_taylor(a.u,'sin')[1])

def cos(a):
    a=asI(a)
    if a.l < D('-1.5') or a.u > D('1.5'):return I(-1,1)
    e1=point_taylor(a.l,'cos');e2=point_taylor(a.u,'cos')
    return I(min(e1[0],e2[0]),D(1) if a.l<=0<=a.u else max(e1[1],e2[1]))

def token_numeric(node,source):
    if isinstance(node,ast.Constant):return D(ast.get_source_segment(source,node))
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -token_numeric(node.operand,source)
    raise ValueError(ast.dump(node)[:200])

def exact_case(name):
    source=Path(__file__).resolve().parent.parent.joinpath('archived_study/deps/prior/verify_power.py').read_text()
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cases')
    mapping={}
    for a in fn.body:
        if isinstance(a,ast.Assign):
            k=a.targets[0]
            if isinstance(k,ast.Name) and k.id in ['b9','e9','b14','e14']:
                mapping[k.id]=[[token_numeric(c,source) for c in r.elts] for r in a.value.args[0].elts]
    ret=next(n.value for n in fn.body if isinstance(n,ast.Return))
    tup=ret.values[0 if name=='IEEE9' else 1]
    vm={int(token_numeric(k,source)):token_numeric(v,source) for k,v in zip(tup.elts[2].keys,tup.elts[2].values)}
    pg={int(token_numeric(k,source)):token_numeric(v,source) for k,v in zip(tup.elts[3].keys,tup.elts[3].values)}
    return mapping['b9' if name=='IEEE9' else 'b14'],mapping['e9' if name=='IEEE9' else 'e14'],vm,pg

def round_binary_float(x):
    return I(D.from_float(float(x)))

def fpower_network(name,center,radius,rel,inject,B_float,load_scale="1",outage_branch_index=None):
    bus,edges,vm,pg=exact_case(name)
    scale=D(str(load_scale))
    bus=[list(row) for row in bus]
    for row in bus: row[1]*=scale; row[2]*=scale
    if outage_branch_index is not None: edges=[v for k,v in enumerate(edges) if k!=outage_branch_index]
    n=len(bus);ang=[i for i in range(n) if bus[i][0]!=3];pq=[i for i in range(n) if bus[i][0]==1];d=len(ang)+len(pq)
    assert len(center)==d
    radius=D(str(radius));rel=D(str(rel));inject=D(str(inject))
    br=[ZERO for _ in range(n)];bi=[ZERO for _ in range(n)]
    Yr=[[ZERO for j in range(n)] for i in range(n)]
    Yi=[[ZERO for j in range(n)] for i in range(n)]
    for i in range(n):Yr[i][i]+=I(bus[i][3]/D(100));Yi[i][i]+=I(bus[i][4]/D(100))
    def interval_prim(val):
        delta=abs(val)*rel
        return I(val-delta,val+delta)
    for f,t,r,x,bb,tap in edges:
        i=int(f)-1;j=int(t)-1
        rr=interval_prim(r);xx=interval_prim(x);bb=interval_prim(bb)
        tt=interval_prim(tap) if tap!=0 else ONE
        den=rr*rr+xx*xx;g=rr/den;h=-xx/den
        yr_ii=g/(tt*tt);yi_ii=(h+bb/I(2))/(tt*tt)
        yr_ij=-g/tt;yi_ij=-h/tt
        Yr[i][i]+=yr_ii;Yi[i][i]+=yi_ii
        Yr[j][j]+=g;Yi[j][j]+=h+bb/I(2)
        Yr[i][j]+=yr_ij;Yi[i][j]+=yi_ij
        Yr[j][i]+=yr_ij;Yi[j][i]+=yi_ij
    def expand(wide):
        th=[ZERO for _ in range(n)];v=[I(b[5]) for b in bus]
        for k,value in vm.items():v[k-1]=I(value)
        for k,i in enumerate(ang):
            c=D.from_float(float(center[k]));th[i]=I(c-radius,c+radius) if wide else I(c)
        for k,i in enumerate(pq):
            c=D.from_float(float(center[len(ang)+k]));v[i]=I(c-radius,c+radius) if wide else I(c)
        return th,v
    ixang={i:k for k,i in enumerate(ang)}
    ixv={i:len(ang)+k for k,i in enumerate(pq)}
    def f_and_der(wide):
        th,v=expand(wide)
        p=[ZERO for _ in range(n)];q=[ZERO for _ in range(n)]
        dp=[[ZERO for _ in range(d)] for _ in range(n)]
        dq=[[ZERO for _ in range(d)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                # skip zeros based on exact support; uncertain coefficients have Y dependency
                if Yr[i][j].l==0==Yr[i][j].u and Yi[i][j].l==0==Yi[i][j].u: continue
                dt=ZERO if i==j else th[i]-th[j];co=cos(dt);si=sin(dt)
                hP=Yr[i][j]*co+Yi[i][j]*si
                hQ=Yr[i][j]*si-Yi[i][j]*co
                pref=v[i]*v[j]
                p[i]+=pref*hP;q[i]+=pref*hQ
                if not wide:continue
                derP=pref*(-Yr[i][j]*si+Yi[i][j]*co)
                derQ=pref*(Yr[i][j]*co+Yi[i][j]*si)
                if i!=j and i in ixang:
                    a=ixang[i];dp[i][a]+=derP;dq[i][a]+=derQ
                if i!=j and j in ixang:
                    a=ixang[j];dp[i][a]-=derP;dq[i][a]-=derQ
                if i in ixv:
                    a=ixv[i];dp[i][a]+=v[j]*hP;dq[i][a]+=v[j]*hQ
                if j in ixv:
                    a=ixv[j];dp[i][a]+=v[i]*hP;dq[i][a]+=v[i]*hQ
        FF=[]
        for i in ang:FF.append(p[i] - I((pg.get(i+1,D(0))-bus[i][1])/D(100)-inject,(pg.get(i+1,D(0))-bus[i][1])/D(100)+inject))
        for i in pq: FF.append(q[i] - I(-bus[i][2]/D(100)-inject,-bus[i][2]/D(100)+inject))
        return FF, ([dp[i] for i in ang]+[dq[i] for i in pq]) if wide else None
    t0=time.time()
    FF,_=f_and_der(False)
    _,JJ=f_and_der(True)
    BF=[[round_binary_float(B_float[i,j]) for j in range(d)] for i in range(d)]
    C=[];u=[];margins=[]
    for i in range(d):
        Bf=sum((BF[i][k]*FF[k] for k in range(d)),ZERO)
        u.append(-Bf)
        arow=[]
        for j in range(d):
            bj=sum((BF[i][k]*JJ[k][j] for k in range(d)),ZERO)
            arow.append((ONE if i==j else ZERO)-bj)
        C.append(arow)
        radiusI=I(-radius,radius)
        K=-Bf + sum((arow[j]*radiusI for j in range(d)),ZERO)
        margins.append(min(K.l+radius,radius-K.u))
    row_sums=[sum((z.mag() for z in row),D(0)) for row in C]
    return {'network':name,'load_scale':str(scale),'outage_branch_index':outage_branch_index,'center':list(map(float,center)),'radius':str(radius),'rel':str(rel),'inject':str(inject),
        'strict_inclusion':all(m>0 for m in margins),'all_rows_contractive':max(row_sums)<1,
        'min_margin':str(min(margins)),'max_majorant_row_sum':str(max(row_sums)),
        'margins':[str(x) for x in margins], 'run_s':round(time.time()-t0,2),
        'precision':PREC,'arithmetic':'directed Decimal intervals with exact-rational Taylor endpoint trigonometry; no mpmath.iv',
        'scope':'root inclusion only; PV/PQ fixed; exact decoded decimal benchmark inputs; candidate and B exact float conversions'}

if __name__=='__main__':
    from scipy.optimize import root
    from scipy.linalg import inv
    here=Path(__file__).parent/'work'
    sys.path.insert(0,str(here))
    from run_robustness import setup
    from deps.prior.verify_power import cases
    output=[]
    # 14-bus base classical root with tested radius
    for name in ['IEEE9','IEEE14']:
        dat=setup(name,1,None)
        r=root(dat['F'],dat['xinit'],jac=dat['J'],options={'maxfev':1800,'xtol':1e-12})
        ctr=r.x
        rad='0.0001' if name=='IEEE9' else '0.0001'
        Bb=inv(dat['J'](ctr))
        result=fpower_network(name,ctr,rad,'0.000001','0.00000001',Bb)
        result['candidate']='classical independently obtained near-root'
        print('FINISHED',name,result['strict_inclusion'],result['min_margin'],result['run_s'],flush=True)
        output.append(result)
        Path(__file__).parent.joinpath('independent_results_working.json').write_text(json.dumps(output,indent=2))
    Path(__file__).parent.joinpath('independent_results.json').write_text(json.dumps(output,indent=2))

def security_case9(center,radius,rel):
    """Outward decimal interval validation of 9-bus voltage/rated-branch constraints.
    Bounds BOTH complex power flow directions at every branch, with primitive
    line uncertainty and fixed PV/PQ. Thermal test uses S^2 to avoid sqrt issues.
    """
    bus,edges,vm,pg=exact_case('IEEE9')
    radius=D(str(radius));rel=D(str(rel));n=len(bus)
    ang=[i for i in range(n) if bus[i][0]!=3]
    pq=[i for i in range(n) if bus[i][0]==1]
    th=[ZERO for i in range(n)];v=[I(b[5]) for b in bus]
    for k,val in vm.items():v[k-1]=I(val)
    for k,i in enumerate(ang):
        c=D.from_float(float(center[k]));th[i]=I(c-radius,c+radius)
    for k,i in enumerate(pq):
        c=D.from_float(float(center[len(ang)+k]));v[i]=I(c-radius,c+radius)
    v_margin=min([z.l-D('0.9') for z in v]+[D('1.1')-z.u for z in v])
    def prim(z):
        h=abs(z)*rel;return I(z-h,z+h)
    def cmult(a,b):return (a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0])
    def cadd(a,b):return (a[0]+b[0],a[1]+b[1])
    def cn(x):return (-x[0],-x[1])
    def conj(a):return(a[0],-a[1])
    vphas=[(v[i]*cos(th[i]),v[i]*sin(th[i])) for i in range(n)]
    rates=[250,250,150,300,150,250,250,250,250]
    branches=[]
    for k,(f,t,r,x,bb,tap) in enumerate(edges):
        i=int(f)-1;j=int(t)-1
        rr=prim(r);xx=prim(x);bb=prim(bb);a=prim(tap) if tap else ONE
        den=rr*rr+xx*xx;y=(rr/den,-xx/den)
        sh=(ZERO,bb/I(2));ysh=cadd(y,sh)
        vf=vphas[i];vt=vphas[j]
        ifrom=cadd(cmult((ysh[0]/(a*a),ysh[1]/(a*a)),vf),cmult((-y[0]/a,-y[1]/a),vt))
        ito=cadd(cmult(ysh,vt),cmult((-y[0]/a,-y[1]/a),vf))
        ss=[]
        for voltage,current in [(vf,ifrom),(vt,ito)]:
            s=cmult(voltage,conj(current))
            # each component abs via mag; apparent power squared upper bound
            ss.append(calc(lambda x,y:x+y,
                calc(lambda x:x*x,s[0].mag(),rounding=CEIL),
                calc(lambda x:x*x,s[1].mag(),rounding=CEIL),rounding=CEIL))
        # rates are in MVA on 100MVA base
        rating=D(rates[k])/D(100)
        margin_sq=rating*rating-max(ss)
        margin_mva=D(rates[k]) - calc(lambda x:x.sqrt(),max(ss),rounding=CEIL)*D(100)
        branches.append(dict(edge=(int(f),int(t)),margin_sq_pu2=str(margin_sq),mva_margin_lower=str(margin_mva),both_ends_pass=margin_sq>0))
    return dict(voltage_min_margin_pu=str(v_margin),all_voltages_safe=v_margin>0,all_branches_safe=all(z['both_ends_pass'] for z in branches),min_branch_margin_mva=str(min(D(z['mva_margin_lower']) for z in branches)),branches=branches)
