import os, sys, pickle, json, glob, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _common as C

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--work_dir", default=None, help="run_jobs.py 的输出目录，缺省 finals_analysis/outputs/")
ap.add_argument("--code_dir", default=None, help="代码包根目录，缺省为本脚本所在代码包")
args = ap.parse_args()
C.quiet(); CODE = C.use_code(args.code_dir); WORK = C.work_dir(args.work_dir)
import numpy as np, pandas as pd
from src.utils import segment_score
from src import model as M
from src.utils import load_config
cfg = load_config(os.path.join(CODE, "config.yaml"))
R = {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in glob.glob(os.path.join(WORK, "res", "*.pkl")) if not os.path.basename(f).startswith("val_")}
U4 = [('2025-10-15', 92), ('2025-10-31', 92), ('2025-11-15', 92), ('2025-11-30', 92)]
out = {}


def pooled(rs, mask_fn=None):
    y, p = [], []
    for r, m in rs:
        y.append(r['y'][m]); p.append(r['pred'][m])
    y = np.concatenate(y); p = np.concatenate(p)
    return segment_score(y, p), len(y)


def sub_of(base, other):
    return np.isin(base['idx'], other['idx'])

# 1) 基线
bs = [R.get('base_%s_%d' % u) for u in U4]
if all(b is not None for b in bs):
    out['base4'] = dict(units=[dict(cut=u[0], H=u[1], score=b['score'], acc=b['acc'], mae_short=b['mae_short'], auc=b['auc'], n=b['n_te']) for u, b in zip(U4, bs)],
                        mean=float(np.mean([b['score'] for b in bs])), acc=float(np.mean([b['acc'] for b in bs])),
                        auc_min=float(min(b['auc'] for b in bs)), auc_max=float(max(b['auc'] for b in bs)))

# 2) 跨区块 / 未见井组：与基线在同一批评估行上比较
def compare(prefix, units):
    rows = []
    for u in units:
        o = R.get('%s_%s_%d' % (prefix, u[0], u[1])); b = R.get('base_%s_%d' % u)
        if o is None or b is None:
            return None
        m = sub_of(b, o)
        assert m.sum() == len(o['y']) and (b['idx'][m] == o['idx']).all()
        rows.append((b, m, o))
    same, n = pooled([(b, m) for b, m, o in rows])
    cross, _ = pooled([(o, np.ones(len(o['y']), bool)) for b, m, o in rows])
    per = [dict(same=segment_score(b['y'][m], b['pred'][m])['score'], cross=o['score'], same_acc=segment_score(b['y'][m], b['pred'][m])['acc'], cross_acc=o['acc'],
                cross_auc=o['auc'], n=len(o['y']), n_tr=o['n_tr']) for b, m, o in rows]
    return dict(n=n, same=same, cross=cross, per=per,
                same_mean=float(np.mean([x['same'] for x in per])), cross_mean=float(np.mean([x['cross'] for x in per])),
                same_acc_mean=float(np.mean([x['same_acc'] for x in per])), cross_acc_mean=float(np.mean([x['cross_acc'] for x in per])),
                n_tr_min=int(min(x['n_tr'] for x in per)), n_tr_max=int(max(x['n_tr'] for x in per)))

out['b1to2'] = compare('b1to2', U4)
out['b2to1'] = compare('b2to1', U4)

fr = []
ok = True
for k in range(3):
    o = R.get('fold%d_2025-11-30_92' % k); b = R.get('base_2025-11-30_92')
    if o is None or b is None:
        ok = False; break
    fr.append((b, sub_of(b, o), o))
if ok:
    same, n = pooled([(b, m) for b, m, o in fr]); cross, _ = pooled([(o, np.ones(len(o['y']), bool)) for b, m, o in fr])
    out['folds'] = dict(n=n, same=same, cross=cross)

# 3) 轻量档
lite = {}
for name in ['t200', 't100l5', 't50l10']:
    rs = [R.get('lite_%s_%s_%d' % (name, u[0], u[1])) for u in U4]
    if all(r is not None for r in rs):
        lite[name] = dict(mean=float(np.mean([r['score'] for r in rs])), acc=float(np.mean([r['acc'] for r in rs])),
                          nodes=float(np.mean([r['nodes'] for r in rs])), et_time=float(np.mean([r['times']['et'] for r in rs])),
                          units=[r['score'] for r in rs])
out['lite'] = lite
if all(b is not None for b in bs):
    out['base_nodes'] = float(np.mean([b['nodes'] for b in bs])); out['base_et_time'] = float(np.mean([b['times']['et'] for b in bs]))

# 4) 特征族消融
abl = {}
for name in ['bank', 'bank-rank']:
    rs = [(u, R.get('abl_%s_%s_%d' % (name, u[0], u[1]))) for u in [U4[1], U4[3]]]
    if all(r is not None for _, r in rs):
        abl[name] = dict(mean=float(np.mean([r['score'] for _, r in rs])), acc=float(np.mean([r['acc'] for _, r in rs])), auc=float(np.mean([r['auc'] for _, r in rs])))
if R.get('base_2025-10-31_92') and R.get('base_2025-11-30_92'):
    rs = [R['base_2025-10-31_92'], R['base_2025-11-30_92']]
    abl['all'] = dict(mean=float(np.mean([r['score'] for r in rs])), acc=float(np.mean([r['acc'] for r in rs])), auc=float(np.mean([r['auc'] for r in rs])))
out['abl'] = abl

# 5) 置信度分级
if all(b is not None for b in bs):
    P = np.concatenate([M.calibrate(b['p_et'], cfg) for b in bs]); y = np.concatenate([b['y'] for b in bs]); pr = np.concatenate([b['pred'] for b in bs])
    conf = np.maximum(P, 1 - P); right = ((pr <= 3) == (y <= 3))
    lv = []
    for name, lo, hi in [('高', 0.9, 1.01), ('中', 0.7, 0.9), ('低', 0.0, 0.7)]:
        m = (conf >= lo) & (conf < hi)
        lv.append(dict(level=name, lo=lo, hi=hi, share=float(m.mean()), acc=float(right[m].mean()), n=int(m.sum())))
    out['conf'] = lv
    out['conf_all_acc'] = float(right.mean())

json.dump(out, open(os.path.join(WORK, "numbers_exp.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
def show(k):
    v = out.get(k)
    print(k, json.dumps(v, ensure_ascii=False, default=float)[:1500] if v is not None else None)
for k in ['base4', 'b1to2', 'b2to1', 'folds', 'lite', 'base_nodes', 'base_et_time', 'abl', 'conf', 'conf_all_acc']:
    show(k)
