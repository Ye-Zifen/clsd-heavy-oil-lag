import os, sys, pickle, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _common as C

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data_dir", required=True, help="数据目录，其下为 train/ 与 test/")
ap.add_argument("--detail", required=True, help="predict.py --detail_output 写出的测试期诊断表")
ap.add_argument("--model", default=None, help="标准档模型文件，缺省 models/model.pkl")
ap.add_argument("--work_dir", default=None, help="02_experiments 的输出目录，缺省 finals_analysis/outputs/")
ap.add_argument("--code_dir", default=None, help="代码包根目录，缺省为本脚本所在代码包")
a = ap.parse_args()
C.quiet(); CODE = C.use_code(a.code_dir); WORK = C.work_dir(a.work_dir)
import numpy as np, pandas as pd, joblib
from src import feature as FT, model as M
from src.utils import load_config, segment_score

cfg = load_config(os.path.join(CODE, "config.yaml"))
P = {}
d = pickle.load(open(os.path.join(WORK, "feats.pkl"), "rb")); X = d["Xtr"]; lag = d["lag"].reset_index(drop=True)
y = lag.OPTIMAL_LAG_DAYS.to_numpy()
gi = pd.read_csv(os.path.join(a.data_dir, "train", "well_group_info.csv"), encoding="utf-8-sig")
blk = gi.drop_duplicates("WELL_GROUP_NAME").set_index("WELL_GROUP_NAME").BLOCK
te = pd.read_csv(os.path.join(a.data_dir, "test", "test_optimal_lag_days.csv"), encoding="utf-8-sig")
te["PROD_DATE"] = pd.to_datetime(te.PROD_DATE)

# 1) 训练标签
P["hist"] = np.bincount(y, minlength=46)[:46].tolist()
rows = []
for _, s in lag.sort_values("PROD_DATE").groupby(["WELL_GROUP_NAME", "PROD_INDICATOR"]):
    t = s.PROD_DATE.to_numpy().astype("datetime64[D]").astype(int); sh = (s.OPTIMAL_LAG_DAYS.to_numpy() <= 3)
    if len(t) < 2: continue
    dt = t[None, :] - t[:, None]; iu = np.triu_indices(len(t), 1)
    rows.append(np.stack([dt[iu], (sh[:, None] == sh[None, :])[iu]], 1))
A = np.concatenate(rows)
P["persist"] = [(lo, hi, float(A[(A[:, 0] >= lo) & (A[:, 0] <= hi), 1].mean()), int(((A[:, 0] >= lo) & (A[:, 0] <= hi)).sum()))
                for lo, hi in [(1, 1), (2, 3), (4, 7), (8, 14), (15, 21), (22, 30), (31, 45), (46, 60), (61, 90)]]

# 2) 标签覆盖
dates = pd.date_range(min(lag.PROD_DATE.min(), pd.Timestamp("2025-03-01")), te.PROD_DATE.max(), freq="D")
tg = sorted(te.WELL_GROUP_NAME.unique()); og = sorted(set(lag.WELL_GROUP_NAME.unique()) - set(tg)); groups = tg + og
gidx = {g: k for k, g in enumerate(groups)}; didx = {dd: k for k, dd in enumerate(dates)}
Cm = np.full((len(groups), len(dates)), np.nan)
for (g, dd), v in lag.assign(s=(y <= 3)).groupby(["WELL_GROUP_NAME", "PROD_DATE"]).s.mean().items():
    Cm[gidx[g], didx[dd]] = v
Tm = np.zeros((len(groups), len(dates)), bool)
for g, dd in te[["WELL_GROUP_NAME", "PROD_DATE"]].drop_duplicates().itertuples(index=False):
    Tm[gidx[g], didx[dd]] = True
P["cover"] = dict(C=Cm, T=Tm, n_test_groups=len(tg), dates=dates, test_start=te.PROD_DATE.min())
ym = lag.PROD_DATE.dt.to_period("M").astype(str)
P["monthly"] = lag.assign(ym=ym, blk=lag.WELL_GROUP_NAME.map(blk)).pivot_table(index="ym", columns="blk", values="OPTIMAL_LAG_DAYS", aggfunc="size", fill_value=0)

# 3) 示例序列与滞后谱（G024 日产油量，2026-02-08）
raw = FT.load_data(a.data_dir, include_test=False); bank = FT.SeriesBank(raw)
def example(g, ind, date, back=134):
    t = bank.pos[pd.Timestamp(date)]
    x = np.nan_to_num(bank.INJ[g]); yr = bank.PRO[g][ind]
    m = ((lag.WELL_GROUP_NAME == g) & (lag.PROD_INDICATOR == ind)).to_numpy()
    s = lag[m].sort_values("PROD_DATE")
    cc = ["pl.90.ma0.L%02d" % k for k in range(46)]
    return dict(g=g, ind=ind, date=pd.Timestamp(date), x=x[t - back:t + 1], y=yr[t - back:t + 1], dates=bank.index[t - back:t + 1],
                hm_dates=s.PROD_DATE.to_numpy(), hm=X.loc[s.index, cc].to_numpy(float), hm_y=s.OPTIMAL_LAG_DAYS.to_numpy())
P["ex_short"] = example("G024", "OIL_PROD_DAILY", "2026-02-08")

# 4) 原始日报示例（G024，2025-07-01 至 2026-02-28）
g0, s0, e0 = "G024", pd.Timestamp("2025-07-01"), pd.Timestamp("2026-02-28")
i0, i1 = bank.pos[s0], bank.pos[e0]
P["raw"] = dict(g=g0, dates=bank.index[i0:i1 + 1], inj=bank.INJ[g0][i0:i1 + 1].astype(float),
                prod={k: bank.PRO[g0][k][i0:i1 + 1].astype(float) for k in FT.PROD_INDICATORS})

# 5) 4 个基线单元的前向评估结果
U4 = [("2025-10-15", 92), ("2025-10-31", 92), ("2025-11-15", 92), ("2025-11-30", 92)]
R = [pickle.load(open(os.path.join(WORK, "res", "base_%s_%d.pkl" % u), "rb")) for u in U4]
vs = list(range(4, 46)); sc = []
for v in vs:
    sc.append(float(np.mean([segment_score(r["y"], np.where(r["pred"] >= 4, v, r["pred"]))["score"] for r in R])))
med = float(np.mean([segment_score(r["y"], np.where(r["pred"] >= 4, np.rint(r["lm"]).astype(int), r["pred"]))["score"] for r in R]))
P["vsweep"] = dict(v=vs, score=sc, median_score=med)
Pc = np.concatenate([M.calibrate(r["p_et"], cfg) for r in R]); ys = np.concatenate([(r["y"] <= 3) for r in R]).astype(float)
pred = np.concatenate([r["pred"] for r in R]); yy = np.concatenate([r["y"] for r in R]); idx = np.concatenate([r["idx"] for r in R])
edges = np.linspace(0, 1, 11); rb = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (Pc >= lo) & ((Pc < hi) if hi < 1 else (Pc <= hi))
    rb.append((float(Pc[m].mean()) if m.any() else np.nan, float(ys[m].mean()) if m.any() else np.nan, int(m.sum())))
P["reliab"] = rb
peak = X["pl.90.ma0.mxa"].to_numpy(float)[idx]; right = ((pred <= 3) == (yy <= 3))
pb = [("无定义", np.isnan(peak))] + [("%g–%g" % (lo, hi), (peak >= lo) & (peak < hi)) for lo, hi in [(0, 3), (3, 5), (5, 8), (8, 12)]] + [("≥12", peak >= 12)]
P["peak_acc"] = [(n, float(m.mean()), float(right[m].mean()) if m.any() else np.nan, int(m.sum())) for n, m in pb]
hh = np.concatenate([(lag.PROD_DATE.to_numpy()[np.asarray(r["idx"])] - np.datetime64(u[0])).astype("timedelta64[D]").astype(int) for u, r in zip(U4, R)])
bins = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 92)]
P["horizon_acc"] = [("%d–%d" % b, float(right[(hh >= b[0]) & (hh <= b[1])].mean())) for b in bins]

# 6) 标准档模型中各特征族的重要性与维数
mdl = joblib.load(a.model or os.path.join(CODE, "models", "model.pkl"))
imp = np.mean([e.feature_importances_ for e in mdl["segment"]], 0); cols = mdl["columns"]
shape_sfx = (".d30", ".d10", ".d21", ".d32", ".rise48", ".rise912", ".max03_rel", ".c3_rel", ".c0_rel")
def fam(c):
    if c.startswith("st"): return "注采窗口统计"
    if c.endswith(shape_sfx): return "谱的短段形状"
    if c.split(".")[0] in ("fl", "pl", "tl", "kl", "rl", "rkl"): return "窗内秩相关滞后谱"
    return "互相关配方库"
fs = pd.Series(imp, index=cols)
P["imp_fam"] = fs.groupby(fs.index.map(fam)).sum().to_dict(); P["dim_fam"] = pd.Series(cols).map(fam).value_counts().to_dict()
del mdl

# 7) 测试期诊断表
det = pd.read_csv(a.detail, encoding="utf-8-sig"); det["PROD_DATE"] = pd.to_datetime(det.PROD_DATE)
P["detail"] = det
P["flag_share"] = det.DATA_FLAG.value_counts(normalize=True).to_dict()

# 8) 监测视图所需的日注汽量
raw_all = FT.load_data(a.data_dir, include_test=True); bank_all = FT.SeriesBank(raw_all)
td = pd.date_range(te.PROD_DATE.min(), te.PROD_DATE.max(), freq="D")
P["inj_test"] = {g: np.nan_to_num(bank_all.INJ[g])[[bank_all.pos[t] for t in td]] for g in tg if g in bank_all.INJ}
P["test_dates"] = td

out = os.path.join(WORK, "figure_data.pkl")
pickle.dump(P, open(out, "wb"))
print("已写出 %s" % out)
