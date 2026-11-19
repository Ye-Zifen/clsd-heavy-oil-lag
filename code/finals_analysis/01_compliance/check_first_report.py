import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _common as C

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data_dir", required=True, help="数据目录，其下为 train/ 与 test/")
ap.add_argument("--code_dir", default=None, help="代码包根目录，缺省为本脚本所在代码包")
a = ap.parse_args()
C.quiet(); C.use_code(a.code_dir)
import numpy as np, pandas as pd
from src import feature as FT
data_dir = a.data_dir
total_diff = 0

lab = pd.read_csv(os.path.join(data_dir, "train", "optimal_lag_days.csv"), encoding="utf-8-sig", parse_dates=["PROD_DATE"])
te = pd.read_csv(os.path.join(data_dir, "test", "test_optimal_lag_days.csv"), encoding="utf-8-sig", parse_dates=["PROD_DATE"])
IND = list(FT.PROD_INDICATORS)

def cut(raw, t):
    r = dict(raw)
    r["inj"] = raw["inj"][raw["inj"].PROD_DATE <= t]
    r["prod"] = raw["prod"][raw["prod"].PROD_DATE <= t]
    return r

for scope, inc in (("训练阶段数据范围", False), ("推理阶段数据范围", True)):
    raw = FT.load_data(data_dir, include_test=inc)
    gi, prod = raw["gi"], raw["prod"]
    declared = gi[gi.WELL_PURPOSE == "生产井"].groupby("WELL_GROUP_NAME").WELL_NAME.apply(set).to_dict()
    dual = set(gi[gi.WELL_PURPOSE == "注汽井"].WELL_NAME)
    samples = pd.concat([lab[["WELL_GROUP_NAME", "PROD_DATE"]]] + ([te[["WELL_GROUP_NAME", "PROD_DATE"]]] if inc else []))
    rows = []
    for g, s in prod.groupby("WELL_GROUP_NAME"):
        pw = declared.get(g, set()); pw2 = pw - dual; sel = pw2 if pw2 else pw
        s_sel = s[s.WELL_NAME.isin(sel)]
        d_any = s.PROD_DATE.min()
        d_sel = s_sel.PROD_DATE.min() if len(s_sel) else pd.NaT
        n_before = int(((samples.WELL_GROUP_NAME == g) & (samples.PROD_DATE < d_sel)).sum()) if len(s_sel) else -1
        rows.append((g, len(sel), d_any, d_sel, n_before))
    df = pd.DataFrame(rows, columns=["g", "n_sel", "d_any", "d_sel", "n_samples_before"])
    print("== %s：%d 个有采出记录的井组" % (scope, len(df)))
    print("  选中的生产井在整张表里没有任何记录（会回退）：%d 个" % int(df.d_sel.isna().sum()))
    print("  组内其他井比选中的生产井更早报数：%d 个" % int((df.d_any < df.d_sel).sum()))
    print("  选中的生产井首次报数之前有样本的井组：%d 个，样本 %d 条"
          % (int((df.n_samples_before > 0).sum()), int(df.n_samples_before.clip(lower=0).sum())))
    print("  首次报数日范围：%s ~ %s；注汽、采出记录最早 %s / %s；样本最早 %s"
          % (df.d_sel.min().date(), df.d_sel.max().date(), raw["inj"].PROD_DATE.min().date(),
             prod.PROD_DATE.min().date(), samples.PROD_DATE.min().date()))
    tot = diff = 0
    for when, off in (("前一天", -1), ("当天", 0)):
        n_g = d_v = n_v = n_empty = 0
        for _, r in df.iterrows():
            t = r.d_sel + pd.Timedelta(days=off)
            rc = cut(raw, t)
            if len(rc["inj"]) == 0 and len(rc["prod"]) == 0:
                n_empty += 1; continue
            keys = pd.DataFrame({"WELL_GROUP_NAME": [r.g] * len(IND), "PROD_DATE": [t] * len(IND), "PROD_INDICATOR": IND})
            Xf = FT.build_features(FT.SeriesBank(raw, extra_dates=[t]), keys, verbose=False, n_jobs=1)
            Xc = FT.build_features(FT.SeriesBank(rc, extra_dates=[t]), keys, verbose=False, n_jobs=1)[Xf.columns]
            xa, xb = Xf.to_numpy(float), Xc.to_numpy(float)
            same = (xa == xb) | (np.isnan(xa) & np.isnan(xb))
            n_g += 1; n_v += xa.size; d_v += int((~same).sum())
            if (~same).any():
                print("    不同：%s %s %s，%d 个值" % (r.g, t.date(), when, int((~same).sum())))
        print("  首次报数%s：%d 个井组截断后没有任何注采记录（早于数据起点）；%d 个井组 × 5 个指标，比较 %d 个特征值，不同 %d 个"
              % (when, n_empty, n_g, n_v, d_v), flush=True)
        total_diff += d_v
sys.exit(0 if total_diff == 0 else 1)
