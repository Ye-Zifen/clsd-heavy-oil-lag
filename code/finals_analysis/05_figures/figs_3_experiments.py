import numpy as np, pandas as pd
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from style import *

TRAINC = "#cdd9ec"


def p14_protocol(P, R, E, cfg, out):
    U = R["units8"]
    rows = [(c, h, s, BLUE) for c, h, s in U["select"]] + [(c, h, s, LIGHT) for c, h, s in U["check"]]
    f = fig(1096, 600)
    gs = f.add_gridspec(1, 2, width_ratios=[0.66, 0.34], left=0.15, right=0.975, top=0.82, bottom=0.1, wspace=0.05)
    ax = f.add_subplot(gs[0, 0]); n = len(rows) + 1
    d0 = mdates.date2num(pd.Timestamp("2025-03-01")); end = pd.Timestamp("2026-02-28"); labs = []
    for i, (cut, H, s, colr) in enumerate(rows):
        y = n - 1 - i; cut = pd.Timestamp(cut); e = min(cut + pd.Timedelta(days=H), end)
        ax.barh(y, mdates.date2num(cut) - d0, left=d0, height=0.6, color=TRAINC)
        ax.barh(y, mdates.date2num(e) - mdates.date2num(cut), left=mdates.date2num(cut), height=0.6, color=colr)
        labs.append("%s · %d 天" % (cut.strftime("%m-%d"), H))
    ax.barh(0, mdates.date2num(end) - d0, left=d0, height=0.6, color=TRAINC)
    ax.barh(0, 92, left=mdates.date2num(end), height=0.6, color=AQUA)
    for yy in (0.5, 2.5): ax.axhline(yy, color="#aab2bd", lw=1)
    ax.set_yticks(range(n)); ax.set_yticklabels(["正式提交"] + labs[::-1], color=INK, fontsize=FS - 2); ax.tick_params(axis="y", length=0)
    tk = pd.to_datetime(["2025-03-01", "2025-07-01", "2025-11-01", "2026-03-01"])
    ax.set_xticks(mdates.date2num(tk)); ax.set_xticklabels([t.strftime("%Y-%m") for t in tk])
    ax.set_xlim(d0, mdates.date2num(pd.Timestamp("2026-06-05"))); ax.set_ylim(-0.6, n - 0.4); xgrid(ax)
    ax.legend(handles=[Patch(color=TRAINC, label="训练：切点以前的标签"), Patch(color=BLUE, label="评估（8 个选参单元）"), Patch(color=LIGHT, label="评估（2 个检查单元）"), Patch(color=AQUA, label="测试期")],
              loc="lower left", bbox_to_anchor=(-0.24, 1.085), ncol=4, columnspacing=1.0, handlelength=1.0, handletextpad=0.4, fontsize=FS - 2)
    ax2 = f.add_subplot(gs[0, 1])
    for i, (cut, H, s, colr) in enumerate(rows): ax2.barh(n - 1 - i, s, height=0.6, color=colr)
    ax2.barh(0, U["b_board"], height=0.6, color=AQUA)
    for yy in (0.5, 2.5): ax2.axhline(yy, color="#aab2bd", lw=1)
    ax2.set_yticks([]); ax2.set_xlim(0, 100); ax2.set_xticks([0, 40, 80]); ax2.set_ylim(-0.6, n - 0.4); xgrid(ax2); ax2.set_xlabel("得分")
    return save(f, out, "p14_protocol")


def p15_results(P, R, E, cfg, out):
    S = R["version_steps"]; Mo = R["rule_by_month"]
    f = fig(1096, 600)
    gs = f.add_gridspec(1, 2, width_ratios=[0.6, 0.4], left=0.08, right=0.985, top=0.88, bottom=0.2, wspace=0.24)
    ax = f.add_subplot(gs[0, 0]); x = np.arange(len(S["offline"]))
    ax.plot(x, S["offline"], color=BLUE, marker="o", markersize=10, markeredgecolor="white", markeredgewidth=2, label="离线：8 个前向评估单元平均")
    ax.plot(x[:len(S["a_board"])], S["a_board"], color=ORANGE, marker="o", markersize=10, markeredgecolor="white", markeredgewidth=2, label="A 榜")
    ax.scatter([x[-1]], [S["b_board"]], s=150, marker="D", color=AQUA, edgecolor="white", linewidth=2, zorder=5, label="初赛 B 榜")
    ax.set_xticks(x); ax.set_xticklabels(S["labels"])
    ax.set_xlim(-0.45, len(x) - 0.55); ax.set_ylim(71, 80.5); ax.set_yticks([72, 74, 76, 78, 80]); ygrid(ax); ax.set_ylabel("得分")
    ax.legend(loc="upper left")
    title(ax, "(a) 合规版本逐步提升")
    ax = f.add_subplot(gs[0, 1])
    ax.plot(range(len(Mo["acc"])), Mo["acc"], color=BLUE, marker="o", markersize=10, markeredgecolor="white", markeredgewidth=2)
    ax.set_xticks(range(len(Mo["acc"]))); ax.set_xticklabels(Mo["months"]); ax.set_xlim(-0.4, len(Mo["acc"]) - 0.6)
    ax.set_ylim(0.7, 0.9); ax.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9]); ygrid(ax); ax.set_ylabel("分段准确率")
    title(ax, "(b) 同一条规则逐月变难")
    return save(f, out, "p15_results")


def p16_negatives(P, R, E, cfg, out):
    items = R["negatives"]["items"]; colr = {"特征": BLUE, "模型": THEME, "决策": ORANGE}
    f = fig(1096, 600)
    ax = f.add_axes([0.335, 0.115, 0.645, 0.775]); n = len(items); y = np.arange(n)[::-1]
    ax.barh(y, [it[2] for it in items], height=0.6, color=[colr[it[0]] for it in items])
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels([it[1] for it in items], color=INK); ax.tick_params(axis="y", length=0)
    ax.set_xlim(-2.45, 0.6); ax.set_xticks([-2, -1.5, -1, -0.5, 0, 0.5]); ax.set_xticklabels(["−2", "−1.5", "−1", "−0.5", "0", "+0.5"]); xgrid(ax)
    ax.set_ylim(-0.7, n - 0.3); ax.set_xlabel("相对各自基线的离线得分变化（分）")
    ax.legend(handles=[Patch(color=c, label=k) for k, c in colr.items()], loc="lower left", bbox_to_anchor=(-0.5, 1.0), ncol=3, columnspacing=1.4, handlelength=1.2)
    return save(f, out, "p16_negatives")


def p17_generalize(P, R, E, cfg, out):
    G = [(E["b1to2"]["same_acc_mean"], E["b1to2"]["cross_acc_mean"], E["b1to2"]["same_mean"], E["b1to2"]["cross_mean"]),
         (E["b2to1"]["same_acc_mean"], E["b2to1"]["cross_acc_mean"], E["b2to1"]["same_mean"], E["b2to1"]["cross_mean"]),
         (E["folds"]["same"]["acc"], E["folds"]["cross"]["acc"], E["folds"]["same"]["score"], E["folds"]["cross"]["score"])]
    f = fig(1096, 600)
    gs = f.add_gridspec(1, 2, left=0.075, right=0.985, top=0.8, bottom=0.2, wspace=0.2)
    labs = ["区块 2 当新区块\n只用区块 1 训练", "区块 1 当新区块\n只用区块 2 训练", "没见过的井组\n留出 1/3"]
    for k, (ttl, ymax) in enumerate((("(a) 分段准确率", 1.0), ("(b) 得分", 100))):
        ax = f.add_subplot(gs[0, k]); x = np.arange(3)
        ax.bar(x - 0.2, [g[2 * k] for g in G], width=0.36, color=GRAY); ax.bar(x + 0.2, [g[2 * k + 1] for g in G], width=0.36, color=BLUE)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=FS - 3, linespacing=1.35); ax.set_ylim(0, ymax * 1.08); ygrid(ax)
        ax.set_yticks(np.linspace(0, ymax, 5)); title(ax, ttl)
    f.legend(handles=[Patch(color=GRAY, label="训练含这部分数据"), Patch(color=BLUE, label="训练完全不含这部分数据")], loc="upper center", bbox_to_anchor=(0.53, 0.985), ncol=2, columnspacing=2, handlelength=1.2)
    return save(f, out, "p17_generalize")


FIGS = [p14_protocol, p15_results, p16_negatives, p17_generalize]
