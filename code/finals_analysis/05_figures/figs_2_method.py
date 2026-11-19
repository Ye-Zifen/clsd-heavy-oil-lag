import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from style import *

SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fe", "#9ec5f4", "#2a78d6", "#0d366b"])


def p08_spectrum(P, R, E, cfg, out):
    ex = P["ex_short"]
    f = fig(1096, 600)
    gs = f.add_gridspec(4, 2, width_ratios=[0.5, 0.5], height_ratios=[0.2, 0.2, 0.1, 0.5], left=0.1, right=0.985, top=0.92, bottom=0.115, hspace=0.3, wspace=0.22)
    d = pd.DatetimeIndex(ex["dates"])[-90:]; x = ex["x"][-90:]; yv = ex["y"][-90:]
    ax = f.add_subplot(gs[0, 0]); ax.plot(d, x, color=ORANGE, drawstyle="steps-post"); ax.set_xticklabels([]); ax.set_ylim(0, np.nanmax(x) * 1.25); ygrid(ax)
    ax.set_ylabel("注汽", rotation=0, ha="right", va="center")
    ax.set_title("(a) 井组 %s 最近 90 天的注采序列" % ex["g"], loc="left", pad=8)
    ax2 = f.add_subplot(gs[1, 0]); ax2.plot(d, yv, color=BLUE); ygrid(ax2); ax2.set_ylim(0, np.nanmax(yv) * 1.3)
    ax2.set_ylabel("产油", rotation=0, ha="right", va="center")
    tk = [d[0], d[30], d[60], d[-1]]
    for a in (ax, ax2): a.set_xticks(tk); a.set_xlim(d[0], d[-1])
    ax2.set_xticklabels([t.strftime("%m-%d") for t in tk]); ax2.get_xticklabels()[-1].set_ha("right")
    hd = pd.DatetimeIndex(ex["hm_dates"])
    c = ex["hm"][list(hd).index(ex["date"])]
    ax3 = f.add_subplot(gs[3, 0]); L = np.arange(46)
    ax3.fill_between(L, c, color=BLUE, alpha=0.12, lw=0); ax3.plot(L, c, color=BLUE)
    ax3.set_xlim(-0.5, 45); ax3.set_ylim(0, np.nanmax(c) * 1.15); ax3.set_xticks([0, 3, 15, 30, 45]); ygrid(ax3)
    ax3.set_xlabel("滞后天数"); ax3.set_ylabel("相关显著性")
    ax3.set_title("(b) %s 的滞后谱" % ex["date"].strftime("%Y-%m-%d"), loc="left", pad=8)
    ax4 = f.add_subplot(gs[:, 1]); m = hd >= pd.Timestamp("2026-01-01")
    H = ex["hm"][m]; yl = ex["hm_y"][m]; hd = hd[m]
    ax4.imshow(np.nan_to_num(H).T, aspect="auto", origin="lower", cmap=SEQ, extent=[-0.5, len(hd) - 0.5, -0.5, 45.5], interpolation="nearest")
    ax4.scatter(np.arange(len(hd)), yl, s=34, color=ORANGE, edgecolor="white", linewidth=0.8, zorder=5)
    ticks = [i for i, t in enumerate(hd) if t.day in (1, 15)]
    ax4.set_xticks(ticks); ax4.set_xticklabels([hd[i].strftime("%m-%d") for i in ticks])
    ax4.set_yticks([0, 3, 15, 30, 45]); ax4.set_ylabel("滞后天数"); ax4.set_xlabel("日期（2026 年）")
    ax4.set_title("(c) 谱随日期的变化", loc="left", pad=8)
    ax4.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor="white", markersize=10, label="真实标签")],
               loc="upper right", frameon=True, facecolor="white", edgecolor="#c9d4e6", framealpha=0.95)
    return save(f, out, "p08_spectrum")


def p09_rules(P, R, E, cfg, out):
    f = fig(1096, 600)
    gs = f.add_gridspec(2, 2, width_ratios=[0.52, 0.48], left=0.3, right=0.985, top=0.9, bottom=0.12, hspace=0.62, wspace=0.52)
    ax = f.add_subplot(gs[:, 0]); lad = R["rule_ladder"]["items"]; yy = np.arange(len(lad))[::-1]
    ax.barh(yy, [v for _, v in lad], height=0.56, color=[GRAY] + [BLUE] * (len(lad) - 1))
    ax.set_yticks(yy); ax.set_yticklabels([n for n, _ in lad], color=INK); ax.set_xlim(0.5, 0.9); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9]); xgrid(ax)
    ax.tick_params(axis="y", length=0); ax.set_xlabel("分段准确率")
    ax.set_title("(a) 单条规则逐步改进", loc="left", pad=10, x=-0.72)
    ax = f.add_subplot(gs[0, 1]); W = R["rule_window"]
    ax.plot(W["window_days"], W["acc"], color=BLUE, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.8)
    ax.set_xticks([60, 80, 90, 100, 120]); ax.set_ylim(0.58, 0.82); ax.set_yticks([0.6, 0.7, 0.8]); ygrid(ax); ax.set_xlabel("相关窗口长度（天）")
    ax.set_title("(b) 窗口长度", loc="left", pad=8)
    ax = f.add_subplot(gs[1, 1]); fm_ = R["rule_fill"]["items"]; y2 = np.arange(len(fm_))[::-1]
    ax.barh(y2, [v for _, v in fm_], height=0.56, color=[BLUE] + [GRAY] * (len(fm_) - 1))
    ax.set_yticks(y2); ax.set_yticklabels([n for n, _ in fm_], color=INK); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.5, 0.86); ax.set_xticks([0.5, 0.6, 0.7, 0.8]); xgrid(ax)
    ax.set_title("(c) 采出缺失的补法", loc="left", pad=8, x=-0.3)
    return save(f, out, "p09_rules")


def p11_features(P, R, E, cfg, out):
    f = fig(1096, 600)
    gs = f.add_gridspec(1, 2, width_ratios=[0.42, 0.58], left=0.08, right=0.985, top=0.88, bottom=0.2, wspace=0.62)
    ax = f.add_subplot(gs[0, 0]); A = E["abl"]
    ax.bar(range(3), [A["bank"]["mean"], A["bank-rank"]["mean"], A["all"]["mean"]], width=0.56, color=[GRAY, BLUE, THEME])
    ax.set_xticks(range(3)); ax.set_xticklabels(["互相关\n配方库", "+ 秩相关\n滞后谱", "+ 谱形状\n+ 窗口统计"])
    ax.set_ylim(0, 92); ax.set_yticks([0, 20, 40, 60, 80]); ygrid(ax); ax.set_ylabel("离线得分")
    title(ax, "(a) 逐步加入特征族")
    ax = f.add_subplot(gs[0, 1])
    fams = ["窗内秩相关滞后谱", "互相关配方库", "谱的短段形状", "注采窗口统计"]
    tot = sum(P["dim_fam"].values()); imp = [P["imp_fam"][k] for k in fams]; dim = [P["dim_fam"][k] / tot for k in fams]
    y = np.arange(len(fams))[::-1]
    ax.barh(y + 0.19, imp, height=0.34, color=BLUE, label="重要性占比")
    ax.barh(y - 0.19, dim, height=0.34, color=GRAY, label="维数占比")
    ax.set_yticks(y); ax.set_yticklabels(fams, color=INK); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.75); ax.set_xticks([0, 0.2, 0.4, 0.6]); ax.set_xticklabels(["0", "20%", "40%", "60%"]); xgrid(ax)
    ax.legend(loc="lower right", handlelength=1.2)
    ax.set_title("(b) 各族在分段分类器中的重要性", loc="left", pad=10, x=-0.52)
    return save(f, out, "p11_features")


def p12_causal_fill(P, R, E, cfg, out):
    f = fig(1096, 600)
    gs = f.add_gridspec(1, 2, width_ratios=[0.5, 0.5], left=0.03, right=0.975, top=0.88, bottom=0.27, wspace=0.52)
    ax = f.add_subplot(gs[0, 0]); ax.set_xlim(0, 100); ax.set_ylim(0, 10)
    for s in ("left", "top", "right"): ax.spines[s].set_visible(False)
    ax.set_yticks([]); ax.set_xticks([])
    ox = np.array([4, 11, 18, 25, 32, 39, 46]); oy = np.array([4.6, 5.8, 5.1, 6.6, 6.0, 7.1, 6.3])
    ax.plot(ox, oy, color=INK, lw=2.2); ax.scatter(ox, oy, s=90, color=INK, edgecolor="white", linewidth=1.8, zorder=5)
    ax.plot([46, 96], [6.3, 2.6], color=MUTED, lw=2.4, ls=(0, (5, 4))); ax.scatter([96], [2.6], s=110, facecolor="white", edgecolor=MUTED, linewidth=2.4, zorder=5)
    ax.plot([46, 80], [6.3, 6.3], color=ORANGE, lw=3.4)
    ax.plot([46, 80], [6.3, 4.9], color=BLUE, lw=3.4)
    ax.plot([80, 80], [0, 8.6], color=INK, lw=1.8)
    ax.legend(handles=[Line2D([], [], color=INK, lw=2.2, marker="o", markersize=7, label="已有观测"),
                       Line2D([], [], color=ORANGE, lw=3.4, label="平直延续"),
                       Line2D([], [], color=BLUE, lw=3.4, label="小斜率单调延续（本方案）"),
                       Line2D([], [], color=MUTED, lw=2.4, ls=(0, (5, 4)), label="线性连到下一次观测"),
                       Line2D([], [], color=INK, lw=1.8, label="预测日")],
              loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=2, handlelength=1.6, labelspacing=0.35, columnspacing=0.8, handletextpad=0.4, fontsize=FS - 4)
    title(ax, "(a) 预测日采出缺失时怎么补")
    ax = f.add_subplot(gs[0, 1]); G = R["causal_fill"]; y = np.arange(len(G["groups"]))[::-1]
    ax.barh(y + 0.19, G["monotone"], height=0.34, color=BLUE, label="因果补齐后")
    ax.barh(y - 0.19, G["flat"], height=0.34, color=GRAY, label="平直延续")
    ax.set_yticks(y); ax.set_yticklabels(G["groups"], color=INK); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); xgrid(ax); ax.set_xlabel("分段准确率")
    ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.2), ncol=2, handlelength=1.2, columnspacing=1.4)
    ax.set_title("(b) 修正前后的分段准确率", loc="left", pad=10, x=-0.42)
    return save(f, out, "p12_causal_fill")


def p13_decision(P, R, E, cfg, out):
    f = fig(1096, 600)
    gs = f.add_gridspec(1, 2, width_ratios=[0.46, 0.54], left=0.085, right=0.985, top=0.88, bottom=0.15, wspace=0.26)
    ax = f.add_subplot(gs[0, 0]); rb = P["reliab"]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, ls=(0, (5, 4)), label="理想校准")
    ax.plot([r[0] for r in rb], [r[1] for r in rb], color=BLUE, marker="o", markersize=10, markeredgecolor="white", markeredgewidth=2, label="本方案")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(color=GRID, linewidth=1); ax.set_axisbelow(True); ax.legend(loc="lower right")
    ax.set_xlabel("模型给出的 P(短段)"); ax.set_ylabel("实际为短段的比例")
    title(ax, "(a) 概率校准")
    ax = f.add_subplot(gs[0, 1])
    pa = cfg["decision"]["p_assume"]; ws, wl = 0.3 * 30.0 / pa, 0.1 * 2.0 / (1.0 - pa)
    p4 = np.array([0.55, 0.15, 0.10, 0.20]); lm = 25.0; v = np.arange(0, 11)
    es = np.array([(p4 * np.abs(k - np.arange(4))).sum() for k in v])
    for Pp, colr in ((0.2, GRAY), (0.4, BLUE), (0.8, THEME)):
        cst = Pp * (60 * (v > 3) + ws * es) + (1 - Pp) * (60 * (v <= 3) + wl * np.abs(v - lm))
        ax.plot(v, cst, color=colr, marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5, label="P(短段) = %.1f" % Pp)
    ax.set_xlim(-0.5, 10.5); ax.set_xticks(range(0, 11, 2)); ax.set_ylim(0, 210); ygrid(ax)
    ax.set_xlabel("输出的滞后天数"); ax.set_ylabel("期望代价（× 1/N 分）"); ax.legend(loc="upper left")
    title(ax, "(b) 各输出天数的期望代价")
    return save(f, out, "p13_decision")


FIGS = [p08_spectrum, p09_rules, p11_features, p12_causal_fill, p13_decision]
