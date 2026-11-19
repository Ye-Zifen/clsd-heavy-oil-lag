import os, sys, glob, json, pickle, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _common as C

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--work_dir", default=None, help="02_experiments 的输出目录，缺省 finals_analysis/outputs/")
ap.add_argument("--code_dir", default=None, help="代码包根目录，缺省为本脚本所在代码包")
args = ap.parse_args()
C.quiet(); CODE = C.use_code(args.code_dir); WORK = C.work_dir(args.work_dir)
import numpy as np, pandas as pd
from src import model as M
from src.utils import load_config, segment_score

cfg = load_config(os.path.join(CODE, "config.yaml"))
U4 = [("2025-10-15", 92), ("2025-10-31", 92), ("2025-11-15", 92), ("2025-11-30", 92)]
files = sorted(f for f in glob.glob(os.path.join(WORK, "res", "*.pkl")) if not os.path.basename(f).startswith("val_"))
if not files:
    raise SystemExit("没有找到 %s/res/*.pkl，请先运行 02_experiments 的 cache_features.py 与 run_jobs.py" % WORK)
R = {os.path.basename(f)[:-4]: pickle.load(open(f, "rb")) for f in files}
out = {}

# (1) 盈亏点
pa = cfg["decision"]["p_assume"]
w_short, w_long = 0.3 * 30.0 / pa, 0.1 * 2.0 / (1.0 - pa)
out["breakeven"] = dict(p_assume=pa, w_short=w_short, w_long=w_long, p_star=w_long / (w_short + w_long),
                        p_floor=cfg["calibration"]["p_floor"])
print("(1) 单样本代价：真短段每差 1 天 %.2f，真长段每差 1 天 %.3f；长段多报 1 天的盈亏点 P(短段) = %.2f%%；概率下限 %.2f"
      % (w_short, w_long, 100 * out["breakeven"]["p_star"], cfg["calibration"]["p_floor"]))

# (2) 长段中位数的作用
n = mism = c4 = c45 = 0
for name, r in R.items():
    P = M.calibrate(r["p_et"], cfg)
    base = M.emit(P, r["p4"], r["lm"], pa)
    mism += int((base != r["pred"]).sum())
    c4 += int((M.emit(P, r["p4"], np.full_like(r["lm"], 4.0), pa) != base).sum())
    c45 += int((M.emit(P, r["p4"], np.full_like(r["lm"], 45.0), pa) != base).sum())
    n += len(base)
out["long_median"] = dict(n_jobs=len(R), n_rows=n, recompute_mismatch=mism, changed_if_4=c4, changed_if_45=c45)
print("(2) %d 个任务、%d 条预测：重新决策与保存的预测不同 %d 行；长段中位数换成 4 后变化 %d 行，换成 45 后变化 %d 行"
      % (len(R), n, mism, c4, c45))

# (3) 长段统一输出 v 天
B = [R.get("base_%s_%d" % u) for u in U4]
if all(b is not None for b in B):
    sweep = {}
    for v in range(4, 46):
        s = []
        for r in B:
            p = r["pred"].copy(); p[p >= 4] = v; s.append(segment_score(r["y"], p)["score"])
        sweep[v] = float(np.mean(s))
    s = []
    for r in B:
        p = r["pred"].copy(); m = p >= 4; p[m] = np.rint(r["lm"][m]).astype(int); s.append(segment_score(r["y"], p)["score"])
    out["long_value_sweep"] = dict(score_by_v=sweep, actual=float(np.mean([r["score"] for r in B])), median_instead=float(np.mean(s)))
    print("(3) 4 个基线单元平均：实际 %.2f；长段统一报 4 / 6 / 10 / 20 / 45 天：%s；改报长段中位数：%.2f"
          % (out["long_value_sweep"]["actual"], " / ".join("%.2f" % sweep[v] for v in (4, 6, 10, 20, 45)), out["long_value_sweep"]["median_instead"]))

    # (4) 时间漂移
    feats = os.path.join(WORK, "feats.pkl")
    if os.path.exists(feats):
        d = pickle.load(open(feats, "rb")); lag = d["lag"].reset_index(drop=True); X = d["Xtr"]
        hh, ok, peak = [], [], []
        for u, r in zip(U4, B):
            idx = np.asarray(r["idx"])
            hh.append((lag.PROD_DATE.to_numpy()[idx] - np.datetime64(u[0])).astype("timedelta64[D]").astype(int))
            ok.append((r["pred"] <= 3) == (r["y"] <= 3))
            peak.append(X["pl.90.ma0.mxa"].to_numpy(float)[idx])
        hh, ok, peak = np.concatenate(hh), np.concatenate(ok), np.concatenate(peak)
        bins = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 92)]
        out["horizon_acc"] = [dict(days="%d-%d" % b, acc=float(ok[(hh >= b[0]) & (hh <= b[1])].mean()), n=int(((hh >= b[0]) & (hh <= b[1])).sum())) for b in bins]
        print("(4) 离训练截止日的天数与分段准确率：" + "；".join("%s 天 %.3f" % (x["days"], x["acc"]) for x in out["horizon_acc"]))

        # (5) 滞后谱峰值分组
        groups = [("无定义", np.isnan(peak))] + [("%g–%g" % (a, b), (peak >= a) & (peak < b)) for a, b in [(0, 3), (3, 5), (5, 8), (8, 12)]] + [("≥12", peak >= 12)]
        err = ~ok
        out["peak_acc"] = [dict(group=g, share=float(m.mean()), acc=float(ok[m].mean()) if m.any() else None,
                                share_of_errors=float(err[m].sum() / max(err.sum(), 1)), n=int(m.sum())) for g, m in groups]
        weak = (peak >= 0) & (peak < 3)
        out["weak_spectrum"] = dict(share=float(weak.mean()), acc=float(ok[weak].mean()), acc_all=float(ok.mean()),
                                    share_of_errors=float(err[weak].sum() / err.sum()))
        print("(5) 弱谱样本（峰值低于 3）占 %.1f%%，分段准确率 %.3f（全体 %.3f），占分段错误的 %.1f%%"
              % (100 * weak.mean(), ok[weak].mean(), ok.mean(), 100 * out["weak_spectrum"]["share_of_errors"]))
        for x in out["peak_acc"]:
            print("    峰值 %-6s 占 %5.1f%%  准确率 %s  占错误 %5.1f%%" % (x["group"], 100 * x["share"], "%.3f" % x["acc"] if x["acc"] is not None else "—", 100 * x["share_of_errors"]))
    else:
        print("(4)(5) 跳过：没有 %s（先运行 cache_features.py）" % feats)
else:
    print("(3)–(5) 跳过：缺少 4 个基线单元（base_*）的结果")

json.dump(out, open(os.path.join(WORK, "decision_checks.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("已写出 %s" % os.path.join(WORK, "decision_checks.json"))
