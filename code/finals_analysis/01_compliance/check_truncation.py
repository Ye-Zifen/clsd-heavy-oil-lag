import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _common as C

DEFAULT_DATES = ["2025-03-20", "2025-07-15", "2025-09-15", "2025-11-15", "2026-01-15", "2026-02-28",
                 "2026-03-01", "2026-03-20", "2026-04-08", "2026-04-20", "2026-04-30", "2026-05-31"]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data_dir", required=True, help="数据目录，其下为 train/ 与 test/")
ap.add_argument("--dates", nargs="*", default=DEFAULT_DATES, help="要核查的预测日（YYYY-MM-DD）")
ap.add_argument("--code_dir", default=None, help="代码包根目录，缺省为本脚本所在代码包")
a = ap.parse_args()
C.quiet(); C.use_code(a.code_dir)
import numpy as np, pandas as pd
from src import feature as FT

lab = pd.read_csv(os.path.join(a.data_dir, "train", "optimal_lag_days.csv"), encoding="utf-8-sig", parse_dates=["PROD_DATE"])
te = pd.read_csv(os.path.join(a.data_dir, "test", "test_optimal_lag_days.csv"), encoding="utf-8-sig", parse_dates=["PROD_DATE"])
KEY = ["WELL_GROUP_NAME", "PROD_DATE", "PROD_INDICATOR"]
raw_tr = FT.load_data(a.data_dir, include_test=False)      # 训练阶段使用的数据范围
raw_all = FT.load_data(a.data_dir, include_test=True)      # 推理阶段使用的数据范围
test_start = te.PROD_DATE.min()


def cut(raw, t):
    r = dict(raw)
    r["inj"] = raw["inj"][raw["inj"].PROD_DATE <= t]
    r["prod"] = raw["prod"][raw["prod"].PROD_DATE <= t]
    return r


tot_rows = tot_vals = tot_diff = 0
for t in [pd.Timestamp(d) for d in a.dates]:
    is_test = t >= test_start
    keys = (te if is_test else lab)
    keys = keys[keys.PROD_DATE == t][KEY].drop_duplicates().reset_index(drop=True)
    if len(keys) == 0:
        print(t.date(), "该日没有样本，跳过"); continue
    raw = raw_all if is_test else raw_tr
    t0 = time.time()
    Xf = FT.build_features(FT.SeriesBank(raw), keys, verbose=False, n_jobs=1)
    Xc = FT.build_features(FT.SeriesBank(cut(raw, t)), keys, verbose=False, n_jobs=1)[Xf.columns]
    x, y = Xf.to_numpy(float), Xc.to_numpy(float)
    same = (x == y) | (np.isnan(x) & np.isnan(y))
    ncol = int((~same).any(axis=0).sum()); nrow = int((~same).any(axis=1).sum())
    tot_rows += len(keys); tot_vals += x.size; tot_diff += int((~same).sum())
    print("%s %s 样本 %d 行 × %d 维：不同的值 %d 个，涉及 %d 行、%d 列（%.0f 秒）"
          % (t.date(), "测试" if is_test else "训练", len(keys), x.shape[1], int((~same).sum()), nrow, ncol, time.time() - t0), flush=True)
print("合计：%d 行，%d 个特征值，截断前后不同的 %d 个" % (tot_rows, tot_vals, tot_diff))
sys.exit(0 if tot_diff == 0 else 1)
