"""Derive data-only diagnostics from archived, fixed-seed experiment. No retraining.
Geometric class compares candidate with the *numerical reference root* and
is not a proof about every possible nonlinear root. """
import csv, json, sys, os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
D=Path(__file__).resolve().parent
ROOT=D
PRIOR=D/'archived_study'
sys.path.insert(0,str(PRIOR));sys.path.insert(0,str(PRIOR/'deps'))
import run_robustness as rb
rows=json.load(open(D/'diagnostics.json'))
archive=json.load(open(PRIOR/'results.json'))
limits={'IEEE9':0.006,'IEEE14':0.010}
out=[];classes={}
for c in archive['cases']:
 d=rb.setup(c['name'],c['load_scale'],c['outage_branch_index']); denom=float(np.linalg.norm(d['b'],np.inf))
 key=(c['name'],c['scenario']); cl=[]
 for r in [t for t in rows if t['network']==c['name'] and t['scenario']==c['scenario'] and t['method']=='VQLS']:
  inside=r['distance_candidate_to_root_inf']<=limits[c['name']]
  label=('accepted_root_enclosure' if r['root_enclosure_archived'] else ('reference_root_outside_all_tested_boxes' if not inside else 'reference_root_inside_largest_box_but_no_inclusion'))
  cl.append(label)
  if r['root_enclosure_archived'] and not inside:raise ValueError('implausible reference-root displacement')
  out.append(dict(network=c['name'],scenario=c['scenario'],seed=r['seed'],reference_root_distance_inf=r['distance_candidate_to_root_inf'],max_radius=limits[c['name']],geometry_diagnostic=label,archive_inclusion=r['root_enclosure_archived'],vqls_residual_inf=r['residual_inf'],rhs_inf=denom,residual_above_rhs=r['residual_inf']>denom,correction_rel_newton_error=r['relative_newton_error_2']))
 classes[c['name']+' '+c['scenario']]={k:cl.count(k) for k in set(cl)}
with open(D/'trial_level_geometry.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
summary=dict(class_counts=classes,observed_inclusion=sum(v['archive_inclusion'] for v in out),outside_reference_box=sum('outside' in v['geometry_diagnostic'] for v in out),inside_largest_failed=sum('inside' in v['geometry_diagnostic'] for v in out),total=len(out),residual_above_rhs=sum(v['residual_above_rhs'] for v in out),radii=limits,classification_caveat='Numerical nominal reference roots; not an exclusion of other nonlinear roots and not a rigorous root nonexistence conclusion')
(D/'geometry_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
# Grouped scatter plot: show every seed, median, and radius lines
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(9.7,5))
scenarios=[('IEEE9','base'),('IEEE9','load20'),('IEEE9','outage_4_5'),('IEEE9','load10_outage_9_4'),('IEEE14','base'),('IEEE14','load20'),('IEEE14','outage_1_2'),('IEEE14','load10_outage_5_6')]
labels=['9 base','9 +20%','9 outage','9 combined','14 base','14 +20%','14 outage','14 combined']
for i,(n,s) in enumerate(scenarios):
 rr=[r for r in out if (r['network'],r['scenario'])==(n,s)];ys=[r['reference_root_distance_inf'] for r in rr]
 ax.scatter([i-0.12,i,i+0.12],ys,s=22,alpha=.8)
 ax.scatter([i],[np.median(ys)],marker='_',s=240,linewidths=2.3,color='black')
ax.axhline(.006,ls='--',lw=1,color='grey');ax.axhline(.010,ls=':',lw=1,color='grey')
ax.text(.02,.006*1.17,'9-bus maximum tested radius',color='grey',fontsize=8)
ax.text(.02,.010*1.17,'14-bus maximum tested radius',color='grey',fontsize=8)
ax.set_yscale('log');ax.set_xticks(range(8),labels,rotation=30,ha='right')
ax.set_ylabel('Mixed-coordinate $\\|\\widehat{x}-x^*\\|_\\infty$ (rad / p.u. numerical units)')
ax.set_title('All 24 fixed-seed VQLS reconstructions vs numerical reference root')
ax.grid(which='both',axis='y',alpha=.22);fig.tight_layout();fig.savefig(D/'all_seed_root_distances.pdf');fig.savefig(D/'all_seed_root_distances.png',dpi=185);plt.close(fig)
