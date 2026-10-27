"""Is a high escape count bad luck or a broken rule?
Keeps each defect's score fixed (mean over the sweep seeds) and redraws the reject pile and the
validation/test split many times. Under the lowest-defect rule the expected escapes per run are
n_test / (n_val + 1).

Example:
    python check_escape_rule.py --data-root data/mvtec --items-dir results/items --runs 400
"""
import argparse, glob
import numpy as np, pandas as pd
from run_pipeline import stratified_split
from second_opinion.data import load_split, read_image, read_mask
from second_opinion.simulate import Item, RuleInspector, build_reject_pile
ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--category',default='bottle')
ap.add_argument('--items-dir',required=True); ap.add_argument('--runs',type=int,default=400); a=ap.parse_args()
S=224
train,test=load_split(a.data_root,a.category)
train_imgs=np.stack([read_image(s.path,S) for s in train])
base=[Item(read_image(s.path,S),read_mask(s.mask_path,S),s.is_defect,s.path.name,"original",s.defect_type) for s in test]
insp=RuleInspector(train_imgs); insp.calibrate_never_miss([i.image for i in base if i.is_defect])
sc=pd.concat([pd.read_csv(f) for f in glob.glob(f'{a.items_dir}/items_seed*.csv')])
mean=sc[sc.is_defect==1].groupby(['defect_type','source']).score.mean().to_dict()
esc=[]
for seed in range(a.runs):
    rng=np.random.default_rng(seed)
    pile,_=build_reject_pile(base,insp,2,rng)
    isd=np.array([p.is_defect for p in pile]); g=np.array([f"{p.defect_type}/{p.source}" for p in pile])
    val=stratified_split(isd,0.5,rng,g)
    s=np.array([mean.get((p.defect_type,p.source),np.nan) for p in pile])
    tl=np.nanmin(s[isd&val]); esc.append(int(((s<tl)&isd&~val).sum()))
esc=np.array(esc)
print('seeds 0-4',esc[:5],'sum',esc[:5].sum())
nt=int((~val&isd).sum()); nv=int((val&isd).sum())
print('mean escapes per run',esc.mean().round(3),'theory',round(nt/(nv+1),3))
