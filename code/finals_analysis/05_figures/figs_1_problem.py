import numpy as np, pandas as pd
import matplotlib
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from style import *

IND = [("LIQ_PROD_DAILY", "日产液量"), ("OIL_PROD_DAILY", "日产油量"), ("WATER_CUT", "含水率"), ("MAX_TUBING_PRES", "油管压力"), ("WH_TEMP", "井口温度")]


def p04_score(P, R, E, cfg, out):
    pa = cfg["decision"]["p_assume"]
    f = fig(1096, 600)
    gs = f.add_gridspec(1, 2, width_ratios=[0.4, 0.6], left=0.085, right=0.985, top=0.88, bottom=0.14, wspace=0.26)
    ax = f.add_subplot(gs[0, 0])
    vals = [60.0, 0.3 * 30.0 / pa, 0.1 * 2.0 / (1.0 - pa)]
    ax.bar(range(3), vals, width=0.55, color=[GRAY, ORANGE, BLUE])
    ax.set_xticks(range(3)); ax.set_xticklabels(["分段判错\n一次", "真短段\n每差 1 天", "真长段\n每差 1 天"]); ax.set_ylim(0, 70); ax.set_yticks([0, 20, 40, 60]); ygrid(ax)
    ax.set_ylabel("单样本代价（× 1/N 分）"); title(ax, "(a) 三种错误的代价")
    ax = f.add_subplot(gs[0, 1]); V = P["vsweep"]
    ax.plot(V["v"], V["score"], color=BLUE, label="判为长段时统一输出 v 天")
    ax.axhline(V["median_score"], color=ORANGE, lw=2, ls=(0, (5, 4)), label="改为输出长段回归中位数")
    ax.set_xlim(3, 46); ax.set_xticks([4, 10, 20, 30, 40, 45]); ax.set_ylim(50, 85); ygrid(ax)
    ax.set_xlabel("判为长段时统一输出的天数 v"); ax.set_ylabel("综合得分"); ax.legend(loc="lower left")
    title(ax, "(b) 判为长段时输出几天最划算")
    return save(f, out, "p04_score")


def p05_raw(P, R, E, cfg, out):
    W = P["raw"]; dates = W["dates"]; x = np.arange(len(dates))
    f = fig(1096, 600)
    gs = f.add_gridspec(6, 1, left=0.135, right=0.985, top=0.9, bottom=0.085, hspace=0.32)
    tk = [i for i, t in enumerate(dates) if t.day == 1]

    def spans(ax, miss):
        k = 0
        while k < len(miss):
            if miss[k]:
                j = k
                while j < len(miss) and miss[j]: j += 1
                ax.axvspan(k - 0.5, j - 0.5, color="#e6e9ee", lw=0); k = j
            else:
                k += 1
    inj = W["inj"]
    ax = f.add_subplot(gs[0]); spans(ax, np.isnan(inj))
    ax.fill_between(x, np.nan_to_num(inj), step="mid", color=ORANGE, alpha=0.2, lw=0); ax.step(x, inj, where="mid", color=ORANGE, lw=2)
    ax.set_ylabel("日注汽量", rotation=0, ha="right", va="center"); axs = [ax]
    for k, nm in IND:
        yv = W["prod"][k]; ax = f.add_subplot(gs[len(axs)]); spans(ax, np.isnan(yv))
        ax.plot(x, yv, color=BLUE, lw=1.8); ax.set_ylabel(nm, rotation=0, ha="right", va="center"); axs.append(ax)
    for i, ax in enumerate(axs):
        ax.set_xlim(-0.5, len(dates) - 0.5); ax.set_xticks(tk); ax.set_xticklabels([dates[j].strftime("%Y-%m") for j in tk] if i == 5 else [])
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
    axs[0].legend(handles=[Patch(color="#e6e9ee", label="当日无记录")], loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2, handlelength=1.2)
    axs[0].set_title("井组 %s 的原始日报" % W["g"], loc="left", pad=10)
    return save(f, out, "p05_raw")


def p06_coverage(P, R, E, cfg, out):
    C = P["cover"]; M = C["C"]; T = C["T"]; dates = C["dates"]; nt = C["n_test_groups"]
    f = fig(1096, 600)
    gs = f.add_gridspec(2, 1, height_ratios=[0.68, 0.32], left=0.085, right=0.985, top=0.9, bottom=0.1, hspace=0.12)
    ax = f.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("ls", [ORANGE, "#f0efec", BLUE])
    img = np.ones(M.shape + (4,)); ok = ~np.isnan(M); img[ok] = cmap(M[ok]); img[T] = matplotlib.colors.to_rgba(AQUA)
    ax.imshow(img, aspect="auto", interpolation="nearest", extent=[0, len(dates), M.shape[0], 0])
    ax.axhline(nt, color=INK, lw=1.2)
    ax.set_yticks([nt / 2, (nt + M.shape[0]) / 2]); ax.set_yticklabels(["测试井组", "其余井组"]); ax.tick_params(axis="y", length=0)
    mt = pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=1), freq="MS"); pos = [(d - dates[0]).days for d in mt]
    ax.set_xticks(pos); ax.set_xticklabels([]); ax.set_xlim(0, len(dates))
    ax.legend(handles=[Patch(color=BLUE, label="训练标签：短段为主"), Patch(color=ORANGE, label="训练标签：长段为主"), Patch(color=AQUA, label="测试期：待预测")],
              loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, columnspacing=1.6, handlelength=1.2)
    ax2 = f.add_subplot(gs[1]); mo = P["monthly"]
    xs = [(pd.Timestamp(m + "-01") - dates[0]).days + 15 for m in mo.index]; bottom = np.zeros(len(mo))
    for col, c in zip(mo.columns, [THEME, LIGHT, GRAY, AQUA]):
        v = mo[col].to_numpy(); ax2.bar(xs, v, width=22, bottom=bottom, color=c, label=str(col)); bottom += v
    ax2.set_xlim(0, len(dates)); ax2.set_xticks(pos[::2]); ax2.set_xticklabels([d.strftime("%Y-%m") for d in mt[::2]])
    ax2.set_ylabel("每月标签数"); ygrid(ax2); ax2.legend(loc="upper right", ncol=2, columnspacing=1.2, handlelength=1.2)
    return save(f, out, "p06_coverage")


def p07_labels(P, R, E, cfg, out):
    f = fig(1096, 600)
    gs = f.add_gridspec(1, 2, width_ratios=[0.58, 0.42], left=0.085, right=0.985, top=0.88, bottom=0.22, wspace=0.24)
    ax = f.add_subplot(gs[0, 0]); h = P["hist"]
    ax.bar(range(46), h, width=0.78, color=[BLUE if i <= 3 else GRAY for i in range(46)])
    ax.set_xlim(-1, 46); ax.set_xticks([0, 3, 6, 15, 25, 35, 45]); ax.set_xlabel("最佳滞后天数"); ax.set_ylabel("训练标签个数"); ygrid(ax)
    ax.legend(handles=[Patch(color=BLUE, label="短段 0–3 天"), Patch(color=GRAY, label="长段 4–45 天")], loc="upper right")
    title(ax, "(a) 标签取值分布")
    ax = f.add_subplot(gs[0, 1]); ps = P["persist"]; x = np.arange(len(ps))
    ax.plot(x, [p[2] for p in ps], color=BLUE, marker="o", markersize=9, markeredgecolor="white", markeredgewidth=2)
    ax.set_xticks(x); ax.set_xticklabels(["%d" % lo if lo == hi else "%d–%d" % (lo, hi) for lo, hi, _, _ in ps], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.4, 1.0); ax.set_xlabel("两个标签相隔的天数"); ax.set_ylabel("分段一致率"); ygrid(ax)
    title(ax, "(b) 同一序列的标签持续性")
    return save(f, out, "p07_labels")


FIGS = [p04_score, p05_raw, p06_coverage, p07_labels]
