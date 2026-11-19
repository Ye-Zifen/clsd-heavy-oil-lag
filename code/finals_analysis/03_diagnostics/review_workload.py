import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _common as C

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--detail", required=True, help="predict.py --detail_output 写出的诊断表")
ap.add_argument("--group", default="G109", help="要列出时间线的井组")
ap.add_argument("--start", default="2026-04-17"); ap.add_argument("--end", default="2026-04-22")
ap.add_argument("--work_dir", default=None, help="输出目录，缺省 finals_analysis/outputs/")
args = ap.parse_args()
C.quiet(); WORK = C.work_dir(args.work_dir)
import numpy as np, pandas as pd

d = pd.read_csv(args.detail, encoding="utf-8-sig")
d["PROD_DATE"] = pd.to_datetime(d.PROD_DATE)
days = np.sort(d.PROD_DATE.unique())
per_day = lambda m: float(d[m].groupby("PROD_DATE").size().reindex(days, fill_value=0).median())
out = dict(n_rows=len(d), n_days=len(days))

# (1) 分级与标记
out["confidence_share"] = d.CONFIDENCE_LEVEL.value_counts(normalize=True).to_dict()
out["flag_share"] = d.DATA_FLAG.value_counts(normalize=True).to_dict()
print("测试期 %d 个样本、%d 个日期" % (len(d), len(days)))
print("(1) 置信度：" + "，".join("%s %.1f%%" % (k, 100 * v) for k, v in out["confidence_share"].items()))
print("    数据标记：" + "，".join("%s %.1f%%" % (k, 100 * v) for k, v in out["flag_share"].items()))

# (2) 复核量
low = d.CONFIDENCE_LEVEL == "低"; flag = d.DATA_FLAG != "正常"
union, both = low | flag, low & flag
out["review"] = dict(union_share=float(union.mean()), union_per_day_median=per_day(union),
                     both_share=float(both.mean()), both_per_day_median=per_day(both))
print("(2) 低置信或带标记：占 %.1f%%，每天中位数 %.0f 个；两者同时满足：占 %.1f%%，每天中位数 %.0f 个"
      % (100 * union.mean(), out["review"]["union_per_day_median"], 100 * both.mean(), out["review"]["both_per_day_median"]))

# (3) 由长转短的切换
d = d.sort_values(["WELL_GROUP_NAME", "PROD_INDICATOR", "PROD_DATE"])
prev = d.groupby(["WELL_GROUP_NAME", "PROD_INDICATOR"]).OPTIMAL_LAG_DAYS.shift()
sw = (prev >= 4) & (d.OPTIMAL_LAG_DAYS <= 3)
out["long_to_short"] = dict(total=int(sw.sum()), per_day_median=per_day(sw))
print("(3) 同一序列由长段转为短段：共 %d 次，每天中位数 %.0f 次" % (sw.sum(), out["long_to_short"]["per_day_median"]))

# (4) 置信度与输出分段
a = (d.OPTIMAL_LAG_DAYS <= 3) & (d.P_SHORT < 0.5)
b = (d.OPTIMAL_LAG_DAYS >= 4) & (d.P_SHORT >= 0.5)
out["conf_vs_output"] = dict(short_with_p_below_half=int(a.sum()), share=float(a.mean()),
                             p_range=[float(d.P_SHORT[a].min()), float(d.P_SHORT[a].max())] if a.any() else None,
                             levels=d.CONFIDENCE_LEVEL[a].value_counts().to_dict(), long_with_p_above_half=int(b.sum()))
print("(4) 报短段但 P(短段) < 0.5：%d 行（%.1f%%），P 在 %s，置信度等级 %s；报长段但 P(短段) ≥ 0.5：%d 行"
      % (a.sum(), 100 * a.mean(), "%.2f–%.2f" % tuple(out["conf_vs_output"]["p_range"]) if a.any() else "—",
         out["conf_vs_output"]["levels"], b.sum()))

# (5) 每周占比
wk = ((d.PROD_DATE - d.PROD_DATE.min()).dt.days // 7).clip(upper=12)
g = d.assign(wk=wk, low=low.loc[d.index], flag=flag.loc[d.index]).groupby("wk").agg(
    start=("PROD_DATE", "min"), low=("low", "mean"), flag=("flag", "mean"), n=("P_SHORT", "size"))
out["weekly"] = [dict(week=int(k), start=str(r.start.date()), low=float(r.low), flag=float(r.flag), n=int(r.n)) for k, r in g.iterrows()]
print("(5) 每周低置信 / 带标记占比：" + "；".join("%s %.0f%%/%.0f%%" % (x["start"][5:], 100 * x["low"], 100 * x["flag"]) for x in out["weekly"]))

# (6) 单井组时间线
m = (d.WELL_GROUP_NAME == args.group) & (d.PROD_DATE >= args.start) & (d.PROD_DATE <= args.end)
tl = d[m]
if len(tl):
    piv = tl.pivot_table(index="PROD_DATE", columns="PROD_INDICATOR", values="OPTIMAL_LAG_DAYS").astype(int)
    ps = tl.pivot_table(index="PROD_DATE", columns="PROD_INDICATOR", values="P_SHORT")
    print("(6) %s %s 至 %s 的输出（天）：" % (args.group, args.start, args.end))
    print(piv.to_string())
    print("    同期 P(短段)：")
    print(ps.round(2).to_string())
    out["timeline"] = dict(group=args.group, output=piv.reset_index().astype(str).to_dict("records"),
                           p_short=ps.round(4).reset_index().astype(str).to_dict("records"))
else:
    print("(6) %s 在 %s 至 %s 没有样本" % (args.group, args.start, args.end))

json.dump(out, open(os.path.join(WORK, "review_workload.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("已写出 %s" % os.path.join(WORK, "review_workload.json"))
