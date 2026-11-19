import numpy as np
from matplotlib.patches import Patch
from style import *

def p18_runtime(P, R, E, cfg, out):
    T = R["runtime"]
    f = fig(1096, 600)
    gs = f.add_gridspec(2, 1, height_ratios=[0.7, 0.3], left=0.3, right=0.955, top=0.9, bottom=0.2, hspace=0.62)
    ax = f.add_subplot(gs[0])
    it = [("初赛版本：单线程全流程", T["single_full"], GRAY), ("决赛版本：2 核并行全流程", T["full_2core"], THEME), ("轻量档：2 核全流程", T["lite_2core"], BLUE),
          ("只推理（11,020 个样本）", T["infer"], BLUE), ("只推理单日（150 个样本）", T["infer_oneday"], BLUE)]
    y = np.arange(len(it))[::-1]
    ax.barh(y, [v for _, v, _ in it], height=0.56, color=[c for _, _, c in it])
    ax.set_yticks(y); ax.set_yticklabels([nm for nm, _, _ in it], color=INK); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1340); ax.set_xticks([0, 300, 600, 900, 1200]); ax.set_xticklabels(["0", "5 分钟", "10 分钟", "15 分钟", "20 分钟"]); xgrid(ax)
    ax.set_title("(a) 实测运行时间（2 核 CPU，无 GPU）", loc="left", pad=10, x=-0.42)
    ax = f.add_subplot(gs[1]); tot = T["full_2core"]
    seg = [(n, v) for n, v in T["stages_full_2core"]]; seg.append(("读写与其他", tot - sum(v for _, v in seg))); x0 = 0
    for (nm, v), c in zip(seg, [BLUE, THEME, LIGHT, GRAY]):
        ax.barh(0, v, left=x0, height=0.6, color=c, label=nm); x0 += v
    ax.set_yticks([0]); ax.set_yticklabels(["决赛版本全流程"], color=INK); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, tot * 1.02); ax.set_ylim(-0.42, 0.75); ax.set_xticks([0, 200, 400, 600]); ax.set_xticklabels(["0", "200 秒", "400 秒", "600 秒"])
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.45), ncol=4, columnspacing=1.0, handlelength=1.0, fontsize=FS - 3)
    ax.set_title("(b) 时间花在哪里", loc="left", pad=10, x=-0.42)
    return save(f, out, "p18_runtime")


def p19_cost(P, R, E, cfg, out):
    names = ["标准档\n400 棵 × 2", "试验档 A\n200 棵 × 2", "试验档 B\n100 棵 × 2", "轻量档\n50 棵 × 2"]
    mb = R["runtime"]["model_mb"]; size = [mb["标准档"], mb["试验档 A"], mb["试验档 B"], mb["轻量档"]]
    L = E["lite"]
    tm = [E["base_et_time"], L["t200"]["et_time"], L["t100l5"]["et_time"], L["t50l10"]["et_time"]]
    sc = [E["base4"]["mean"], L["t200"]["mean"], L["t100l5"]["mean"], L["t50l10"]["mean"]]
    cols = [THEME, GRAY, GRAY, BLUE]
    f = fig(1096, 600)
    gs = f.add_gridspec(1, 3, left=0.16, right=0.975, top=0.86, bottom=0.14, wspace=0.14)
    y = np.arange(4)[::-1]
    for k, (vals, ttl, xm, xl) in enumerate(((size, "(a) 模型文件", 300, "MB"), (tm, "(b) 分类器训练用时", 300, "秒"))):
        ax = f.add_subplot(gs[0, k]); ax.barh(y, vals, height=0.56, color=cols)
        ax.set_xlim(0, xm); xgrid(ax); ax.set_yticks(y); ax.tick_params(axis="y", length=0)
        ax.set_yticklabels(names if k == 0 else [], color=INK, linespacing=1.35)
        ax.set_xticks([0, 100, 200, 300]); ax.set_xlabel(xl); title(ax, ttl)
    ax = f.add_subplot(gs[0, 2])
    for y0, v, c in zip(y, sc, cols):
        ax.plot([79.2, v], [y0, y0], color="#d5dae0", lw=2, zorder=1); ax.scatter([v], [y0], s=190, color=c, edgecolor="white", linewidth=2, zorder=3)
    ax.set_xlim(79.2, 80.5); ax.set_xticks([79.5, 80.0, 80.5]); ax.set_ylim(-0.6, 3.6); ax.set_yticks(y); ax.set_yticklabels([]); ax.tick_params(axis="y", length=0); xgrid(ax)
    ax.set_xlabel("得分"); title(ax, "(c) 离线得分")
    return save(f, out, "p19_cost")


def p20_confidence(P, R, E, cfg, out):
    """(a) 置信度分级的样本占比与分段准确率；(b) 按滞后谱峰值分组的分段准确率；(c) 测试期数据标记的占比"""
    f = fig(1096, 600)
    ax = f.add_axes([0.075, 0.24, 0.34, 0.64]); C = E["conf"]; x = np.arange(3)
    ax.bar(x - 0.2, [c["share"] for c in C], width=0.36, color=GRAY, label="样本占比")
    ax.bar(x + 0.2, [c["acc"] for c in C], width=0.36, color=BLUE, label="分段准确率")
    ax.set_xticks(x); ax.set_xticklabels(["高\n≥ 0.9", "中\n0.7–0.9", "低\n< 0.7"], linespacing=1.35); ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"]); ygrid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.2, columnspacing=1.4)
    title(ax, "(a) 置信度分级")
    ax = f.add_axes([0.51, 0.6, 0.452, 0.28]); pk = P["peak_acc"]; x = np.arange(len(pk))
    ax.bar(x, [p[2] for p in pk], width=0.6, color=[GRAY] + [ORANGE] + [BLUE] * (len(pk) - 2))
    ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pk]); ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.5, 1.0]); ygrid(ax)
    ax.set_xlabel("滞后谱峰值强度"); ax.set_ylabel("分段准确率")
    title(ax, "(b) 按谱峰值分组")
    ax = f.add_axes([0.51, 0.24, 0.452, 0.115]); fs_ = P["flag_share"]; x0 = 0
    for nm, c in (("正常", BLUE), ("弱相关", ORANGE), ("采出观测不足", GRAY), ("滞后谱无定义", "#8a939e")):
        v = fs_.get(nm, 0.0); ax.barh(0, v, left=x0, height=0.6, color=c, label=nm); x0 += v
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.4, 0.8); ax.set_yticks([]); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=4, columnspacing=1.0, handlelength=1.0, fontsize=FS - 3)
    ax.set_title("(c) 测试期样本的数据标记", loc="left", pad=8)
    return save(f, out, "p20_confidence")


FIGS = [p18_runtime, p19_cost, p20_confidence]
