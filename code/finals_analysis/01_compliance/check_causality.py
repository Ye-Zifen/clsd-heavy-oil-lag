import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _common as C

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data_dir", required=True, help="数据目录，其下为 train/ 与 test/")
ap.add_argument("--code_dir", default=None, help="代码包根目录，缺省为本脚本所在代码包")
ap.add_argument("--n_jobs", type=int, default=1, help="特征构造的进程数，只影响速度")
a = ap.parse_args()
C.quiet(); C.use_code(a.code_dir)
import numpy as np
from src.feature import load_data, SeriesBank, build_features

t0 = time.time()
raw_tr = load_data(a.data_dir, include_test=False)
keys = raw_tr["lag_train"][["WELL_GROUP_NAME", "PROD_DATE", "PROD_INDICATOR"]]
A = build_features(SeriesBank(raw_tr), keys, verbose=False, n_jobs=a.n_jobs)
raw_all = load_data(a.data_dir, include_test=True)
B = build_features(SeriesBank(raw_all), keys, verbose=False, n_jobs=a.n_jobs)
assert list(A.columns) == list(B.columns)
x, y = A.to_numpy(np.float64), B.to_numpy(np.float64)
same = (x == y) | (np.isnan(x) & np.isnan(y))
bad_cols = [c for c, ok in zip(A.columns, same.all(0)) if not ok]
print("训练样本 %d 行 × %d 维，共 %d 个特征值" % (x.shape[0], x.shape[1], x.size))
print("两种数据范围下取值相同的比例：%.6f；有差异的特征列：%d %s" % (same.mean(), len(bad_cols), bad_cols[:10]))
print("最大绝对差：%.3g；用时 %.0f 秒" % (float(np.nanmax(np.abs(np.where(same, 0.0, x - y)))), time.time() - t0))
sys.exit(0 if same.all() else 1)
