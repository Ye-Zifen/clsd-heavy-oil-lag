import numpy as np, pandas as pd
import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from style import *

DIV = LinearSegmentedColormap.from_list("ls", [ORANGE, "#f0efec", BLUE])
IND = [("LIQ_PROD_DAILY", "日产液量"), ("OIL_PROD_DAILY", "日产油量"), ("WATER_CUT", "含水率"), ("MAX_TUBING_PRES", "油管压力"), ("WH_TEMP", "井口温度")]


def p21_monitor(P, R, E, cfg, out, g="G109"):
    dates = P["test_dates"]; inj = P["inj_test"][g]
    D = P["detail"]; D = D[D.WELL_GROUP_NAME == g]
    M = np.full((5, len(dates)), np.nan); low = np.zeros((5, len(dates)), bool)
    for i, (k, _) in enumerate(IND):
        s = D[D.PROD_INDICATOR == k]; j = (s.PROD_DATE - dates[0]).dt.days.to_numpy()
        M[i, j] = s.P_SHORT.to_numpy(); low[i, j] = ((s.CONFIDENCE_LEVEL == "低") | (s.DATA_FLAG != "正常")).to_numpy()
    f = fig(1096, 600)
    gs = f.add_gridspec(3, 1, height_ratios=[0.25, 0.5, 0.25], left=0.115, right=0.985, top=0.9, bottom=0.1, hspace=0.62)
    x = np.arange(len(dates)); tk = [i for i, t in enumerate(dates) if t.day == 1] + [len(dates) - 1]
    tl = [dates[i].strftime("%m-%d") for i in tk]
    ax = f.add_subplot(gs[0]); ax.fill_between(x, inj, step="mid", color=ORANGE, alpha=0.18, lw=0); ax.step(x, inj, where="mid", color=ORANGE, lw=2.2)
    ax.set_xlim(-0.5, len(dates) - 0.5); ax.set_ylim(0, max(inj.max() * 1.25, 1)); ax.set_xticks(tk); ax.set_xticklabels([]); ygrid(ax)
    ax.set_title("(a) 井组 %s 的日注汽量" % g, loc="left", pad=8)
    ax = f.add_subplot(gs[1]); img = np.ones(M.shape + (4,)); ok = ~np.isnan(M); img[ok] = DIV(M[ok])
    ax.imshow(img, aspect="auto", interpolation="nearest", extent=[-0.5, len(dates) - 0.5, 5, 0])
    yy, xx = np.where(low); ax.scatter(xx, yy + 0.5, s=9, color=INK, zorder=4, linewidths=0)
    for k in range(1, 5): ax.axhline(k, color="white", lw=2)
    ax.set_yticks(np.arange(5) + 0.5); ax.set_yticklabels([n for _, n in IND], color=INK); ax.tick_params(axis="y", length=0)
    ax.set_xticks(tk); ax.set_xticklabels([]); ax.set_xlim(-0.5, len(dates) - 0.5)
    for s in ("left", "bottom"): ax.spines[s].set_visible(False)
    ax.set_title("(b) 各指标的响应快慢", loc="left", pad=8)
    ax.legend(handles=[Patch(color=BLUE, label="短段：0–3 天"), Patch(color=ORANGE, label="长段：4 天以上"), Patch(color="#f0efec", label="不确定"),
                       Line2D([], [], marker="o", color="none", markerfacecolor=INK, markersize=5, label="建议复核")],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, columnspacing=1.2, handlelength=1.1, handletextpad=0.5, fontsize=FS - 2, borderaxespad=0.2)
    ax = f.add_subplot(gs[2]); ax.bar(x, low.sum(axis=0), width=0.8, color=GRAY)
    ax.set_xlim(-0.5, len(dates) - 0.5); ax.set_ylim(0, 5.5); ax.set_yticks([0, 5]); ygrid(ax)
    ax.set_xticks(tk); ax.set_xticklabels(tl); ax.get_xticklabels()[-1].set_ha("right")
    ax.set_title("(c) 当日建议复核的指标数", loc="left", pad=8)
    return save(f, out, "p21_monitor")


def p23_risk(P, R, E, cfg, out):
    f = fig(1096, 600)
    gs = f.add_gridspec(1, 2, width_ratios=[0.46, 0.54], left=0.085, right=0.985, top=0.88, bottom=0.2, wspace=0.26)
    ax = f.add_subplot(gs[0, 0]); H = P["horizon_acc"]; x = np.arange(len(H))
    ax.plot(x, [h[1] for h in H], color=BLUE, marker="o", markersize=10, markeredgecolor="white", markeredgewidth=2)
    ax.set_xticks(x); ax.set_xticklabels([h[0] for h in H], fontsize=FS - 3)
    ax.set_ylim(0.75, 0.95); ax.set_yticks([0.75, 0.8, 0.85, 0.9, 0.95]); ygrid(ax); ax.set_xlabel("距训练截止日的天数"); ax.set_ylabel("分段准确率")
    title(ax, "(a) 离训练截止越远，越不准")
    D = P["detail"].copy(); D["wk"] = ((D.PROD_DATE - D.PROD_DATE.min()).dt.days // 7).clip(upper=12)
    g = D.groupby("wk").agg(low=("CONFIDENCE_LEVEL", lambda s: (s == "低").mean()), flag=("DATA_FLAG", lambda s: (s != "正常").mean()))
    ax = f.add_subplot(gs[0, 1])
    ax.plot(g.index, g.flag, color=ORANGE, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.6, label="带数据标记")
    ax.plot(g.index, g.low, color=BLUE, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.6, label="低置信度")
    ax.set_xlim(-0.5, 12.5); ax.set_xticks([0, 4, 8, 12]); ax.set_xticklabels(["3 月初", "4 月初", "4 月底", "5 月底"])
    top = max(g.flag.max(), g.low.max()); ax.set_ylim(0, np.ceil(top * 10 + 1) / 10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); ygrid(ax)
    ax.set_xlabel("测试期（按周）"); ax.set_ylabel("样本占比"); ax.legend(loc="upper left")
    title(ax, "(b) 无标签时可监测的两个量")
    return save(f, out, "p23_risk")

FIGS = [p21_monitor, p23_risk]
