import os, sys, time, pickle, json, copy, hashlib, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _common as C

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data_dir", required=True, help="数据目录，其下为 train/ 与 test/")
ap.add_argument("--jobs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs.json"), help="任务列表")
ap.add_argument("--only", nargs="*", default=None, help="只运行这些名字的任务")
ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
ap.add_argument("--work_dir", default=None, help="输出目录，缺省 finals_analysis/outputs/")
ap.add_argument("--code_dir", default=None, help="代码包根目录，缺省为本脚本所在代码包")
args = ap.parse_args()
C.quiet()
CODE = C.use_code(args.code_dir); DATA = args.data_dir; WORK = C.work_dir(args.work_dir)
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from src import model as M
from src.utils import load_config, segment_score

UNITS=[('2025-10-15',92),('2025-10-31',92),('2025-11-15',92),('2025-11-30',92),('2025-11-30',46),('2025-12-15',46),('2025-12-31',46),('2026-01-15',44)]
RECENT=[('2026-01-31',28),('2026-02-14',14)]
RES=os.path.join(WORK,"res")
os.makedirs(RES, exist_ok=True)
_D=None
def data():
    global _D
    if _D is None:
        d=pickle.load(open(os.path.join(WORK,"feats.pkl"),"rb"))
        gi=pd.read_csv(os.path.join(DATA,"train","well_group_info.csv"),encoding="utf-8-sig")
        blk=gi.drop_duplicates("WELL_GROUP_NAME").set_index("WELL_GROUP_NAME").BLOCK
        d['lag']=d['lag'].reset_index(drop=True)
        d['block']=d['lag'].WELL_GROUP_NAME.map(blk).to_numpy()
        _D=d
    return _D

def col_subset(cols, name):
    if name in (None,"all"): return list(cols)
    rank=[c for c in cols if c.split('.')[0] in ('fl','pl','tl','kl','rl','rkl') and not any(c.endswith(s) for s in ('.d30','.d10','.d21','.d32','.rise48','.rise912','.max03_rel','.c3_rel','.c0_rel'))]
    shape=[c for c in cols if any(c.endswith(s) for s in ('.d30','.d10','.d21','.d32','.rise48','.rise912','.max03_rel','.c3_rel','.c0_rel'))]
    stats=[c for c in cols if c.startswith('st.') or c.startswith('st30.') or c.startswith('st60.') or c.startswith('st90.') or c.startswith('st120.')]
    bank=[c for c in cols if c not in set(rank)|set(shape)|set(stats)]
    fam={'bank':bank,'rank':rank,'shape':shape,'stats':stats}
    out=[]
    for k in name.split('+'): out+=fam[k]
    return [c for c in cols if c in set(out)]

def group_fold(g, k):
    return int(hashlib.md5(g.encode()).hexdigest(),16) % k

def run(job, cfg=None, force=False):
    """job: dict(name, cut, H, train=..., test=..., cols=..., et=dict(...), members=[...])"""
    out=os.path.join(RES, job['name']+".pkl")
    if os.path.exists(out) and not force:
        return pickle.load(open(out,"rb"))
    d=data(); X=d['Xtr']; lag=d['lag']; blk=d['block']
    cfg=copy.deepcopy(cfg or load_config(os.path.join(CODE,"config.yaml")))
    cut=pd.Timestamp(job['cut']); H=job['H']
    dts=lag.PROD_DATE
    tr=(dts<=cut).to_numpy(); te=((dts>cut)&(dts<=cut+pd.Timedelta(days=H))).to_numpy()
    g=lag.WELL_GROUP_NAME.to_numpy()
    def filt(spec, base):
        if spec in (None,'all'): return base
        kind,val=spec.split(':',1)
        if kind=='block': return base & (blk==val)
        if kind=='notblock': return base & (blk!=val)
        if kind=='fold':
            k,K=map(int,val.split('/')); f=np.array([group_fold(x,K) for x in g]); return base & (f==k)
        if kind=='notfold':
            k,K=map(int,val.split('/')); f=np.array([group_fold(x,K) for x in g]); return base & (f!=k)
        raise ValueError(spec)
    tr=filt(job.get('train'),tr); te=filt(job.get('test'),te)
    cols=col_subset(list(X.columns), job.get('cols'))
    Xs=X[cols]
    y=lag.OPTIMAL_LAG_DAYS.to_numpy()
    anchor=dts[tr].max()
    w=0.5**((anchor-dts).dt.days.to_numpy()/cfg["training"]["recency_halflife_days"])
    T={}
    vkey="val_%s_%d_%s_%s_%s"%(job['cut'],H,job.get('train','all'),job.get('test','all'),job.get('cols','all'))
    vkey=vkey.replace('/','-').replace(':','-')
    vf=os.path.join(RES,vkey+".pkl")
    if os.path.exists(vf):
        p4,lm=pickle.load(open(vf,"rb"))
    else:
        t=time.time(); fix=cfg["lightgbm_common"]
        ms=tr&(y<=3); clf=lgb.LGBMClassifier(**fix, **cfg["short_value_model"]); clf.fit(Xs[ms], y[ms], sample_weight=w[ms])
        pv=clf.predict_proba(Xs[te]); p4=np.zeros((te.sum(),4))
        for j,k in enumerate(clf.classes_): p4[:,int(k)]=pv[:,j]
        p4/=np.maximum(p4.sum(1,keepdims=True),1e-12)
        ml=tr&(y>=4); reg=lgb.LGBMRegressor(**fix, **cfg["long_value_model"]); reg.fit(Xs[ml], y[ml], sample_weight=w[ml])
        lm=np.clip(reg.predict(Xs[te]),4,45); T['lgb']=time.time()-t
        pickle.dump((p4,lm),open(vf,"wb"))
    sc={k:v for k,v in cfg["segment_classifier"].items() if k not in ("missing_value","random_states","targets")}
    sc.update(job.get('et',{}))
    kinds=job.get('members') or list(zip(cfg["segment_classifier"]["random_states"], cfg["segment_classifier"]["targets"]))
    Xe=Xs.fillna(cfg["segment_classifier"]["missing_value"]).to_numpy(np.float32)
    ps=[]; nodes=0; t=time.time()
    for rs,kind in kinds:
        et=ExtraTreesClassifier(n_jobs=job.get('n_jobs',1), random_state=rs, **sc)
        et.fit(Xe[tr], M._seg_targets(y[tr],kind), sample_weight=w[tr]); et.seg_kind=kind; et.n_jobs=1
        ps.append(M._p_short(et, Xe[te])); nodes+=sum(e.tree_.node_count for e in et.estimators_)
        del et
    T['et']=time.time()-t
    p_et=np.mean(ps,axis=0)
    P=M.calibrate(p_et,cfg)
    pred=M.emit(P,p4,lm,cfg["decision"]["p_assume"])
    yt=y[te]
    s=segment_score(yt,pred)
    try: auc=float(roc_auc_score((yt<=3).astype(int),p_et))
    except Exception: auc=float('nan')
    res=dict(job=job, n_tr=int(tr.sum()), n_te=int(te.sum()), auc=auc, nodes=int(nodes), times=T, idx=np.where(te)[0], p_et=p_et, ps=ps, p4=p4, lm=lm, pred=pred, y=yt, **s)
    pickle.dump(res,open(out,"wb"))
    return res

if __name__=="__main__":
    jobs=json.load(open(args.jobs, encoding="utf-8"))
    if args.only: jobs=[j for j in jobs if j['name'] in set(args.only)]
    for i,job in enumerate(jobs):
        if i%args.nparts!=args.part: continue
        t=time.time(); r=run(job)
        print("%-40s n_tr %6d n_te %5d auc %.4f acc %.4f maeS %.3f maeL %.2f score %.2f  (%.0fs)"%(job['name'],r['n_tr'],r['n_te'],r['auc'],r['acc'],r['mae_short'],r['mae_long'],r['score'],time.time()-t), flush=True)
