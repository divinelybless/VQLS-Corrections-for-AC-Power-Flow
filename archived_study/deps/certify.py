"""Identical primitive-uncertainty interval checks and voltage / branch security.
A failure of strict Krawczyk inclusion is only failure to certify the chosen box.
For IEEE 14 MATPOWER branch thermal limits are zeros, hence no branch claim.
"""
from pathlib import Path
import json,sys,time
import numpy as np
from mpmath import iv
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prior.verify_power import cases
from interval_center_test import run_case, point, ends, absub
iv.dps=45
RATES9=np.array([250.,250.,150.,300.,150.,250.,250.,250.,250.])

def security(name,center,radius,rel=1e-7):
    bus,edges,vmset,pg=cases()[name]
    n=len(bus);ang=np.flatnonzero(bus[:,0]!=3);pq=np.flatnonzero(bus[:,0]==1)
    th=[point(z) for z in np.deg2rad(bus[:,6])]
    v=[point(z) for z in bus[:,5]]
    for k,z in vmset.items():v[k-1]=point(z)
    for k,i in enumerate(ang):th[i]=iv.mpf([float(center[k]-radius),float(center[k]+radius)])
    for k,i in enumerate(pq):v[i]=iv.mpf([float(center[len(ang)+k]-radius),float(center[len(ang)+k]+radius)])
    lo=[ends(z)[0] for z in v];hi=[ends(z)[1] for z in v]
    native_min, native_max=(.9,1.1) if name=='IEEE9' else (.94,1.06)
    natviol=[int(i+1) for i,(a,b) in enumerate(zip(lo,hi)) if a>native_max or b<native_min]
    native_all=bool(all(a>=native_min and b<=native_max for a,b in zip(lo,hi)))
    result={'native_voltage_min':native_min,'native_voltage_max':native_max,
      'native_voltage_all_safe':native_all,'native_certified_violating_buses':natviol,
      'voltage_all_safe_0p9_1p1':bool(min(lo)>=.9 and max(hi)<=1.1),
     'voltage_min_lower':float(min(lo)),'voltage_max_upper':float(max(hi)),'branch_thermal_limited':name=='IEEE9'}
    if name!='IEEE9':
        result['line_security']='not evaluated: no positive rateA limits provided in official 14-bus benchmark'
        return result
    def prim(z):
        zz=float(z);delta=abs(zz)*rel if zz else 0
        return iv.mpf([zz-delta,zz+delta])
    def prod(g,h,xr,xi):return g*xr-h*xi,g*xi+h*xr
    margins=[]
    for k,(f,t,rr,xx,bb,tap) in enumerate(edges):
        i=int(f)-1;j=int(t)-1
        rr=prim(rr);xx=prim(xx);bb=prim(bb); a=prim(tap) if tap else point(1)
        den=rr*rr+xx*xx; g=rr/den;h=-xx/den
        vi_r,vi_i=v[i]*iv.cos(th[i]),v[i]*iv.sin(th[i])
        vj_r,vj_i=v[j]*iv.cos(th[j]),v[j]*iv.sin(th[j])
        ir1,ii1=prod(g/a**2,(h+bb/2)/a**2,vi_r,vi_i)
        ir2,ii2=prod(-g/a,-h/a,vj_r,vj_i)
        ifr,ifi=ir1+ir2,ii1+ii2
        ir3,ii3=prod(g,h+bb/2,vj_r,vj_i)
        ir4,ii4=prod(-g/a,-h/a,vi_r,vi_i)
        itr,iti=ir3+ir4,ii3+ii4
        def upper(vr,vi,ir,ii):
            sr=vr*ir+vi*ii;si=vi*ir-vr*ii
            return np.hypot(absub(sr),absub(si))*100
        sf=upper(vi_r,vi_i,ifr,ifi);st=upper(vj_r,vj_i,itr,iti)
        margins.append(float(RATES9[k]-max(sf,st)))
    result['minimum_line_MVA_margin_lower']=float(min(margins))
    result['line_security']=bool(min(margins)>0)
    return result

def main():
    report=json.loads((HERE/'measurement_and_matching_results.json').read_text())
    for name,case in report['cases'].items():
        bus,edge,vm,pg=cases()[name]
        radii=([1e-4,.0015,.003] if name=='IEEE9' else [1e-4,.0025,.006])
        for k,center in case['centres'].items():
            entries=[]
            for radius in radii:
                tic=time.perf_counter()
                z=run_case(name,bus,edge,vm,pg,rel=1e-7,inject=1e-9,radius=radius,center=center)
                tim=time.perf_counter()-tic
                ok=z['krawczyk_strict_inclusion']
                s=security(name,center,radius) if ok else None
                entries.append({'radius':radius,'strict_inclusion':bool(ok),
                  'majorant_inf':z['interval_majorant_rowsum'],
                  'inclusion_margin':z['krawczyk_inclusion_margin_min'],
                  'interval_seconds':tim,'security':s})
                print(name,k, radius, 'inclusion',ok,'row',round(z['interval_majorant_rowsum'],3),'time',round(tim,2),flush=True)
            case['candidates'][k]['interval_results']=entries
            good=[item for item in entries if item['strict_inclusion'] and item['security'] and item['security']['voltage_all_safe_0p9_1p1'] and (name!='IEEE9' or item['security']['line_security'])]
            case['candidates'][k]['certified_security_at_any_tested_radius']=bool(good)
            case['candidates'][k]['smallest_successful_test_radius']=min((p['radius'] for p in good), default=None)
    (HERE/'certificate_acceptance_results.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
