import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

THEME = "#1d4e9e"     # 版式主色
BLUE = "#2a78d6"      # 数据主色
ORANGE = "#eb6834"    # 数据强调色
GRAY = "#b9c1c9"      # 对照 / 次要
LIGHT = "#86b6ef"     # 浅蓝
AQUA = "#1baf7a"      # 测试期
INK = "#1a1a1a"
MUTED = "#5f6873"
GRID = "#e3e6ea"
TINT = "#e9f0fb"
FS = 17               # 基准字号

CJK_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
             "PingFang SC", "Hiragino Sans GB", "WenQuanYi Micro Hei", "Arial Unicode MS"]


def _pick_font():
    path = os.environ.get("FIG_FONT")
    if path and os.path.exists(path):
        fm.fontManager.addfont(path)
        return fm.FontProperties(fname=path).get_name()
    have = {f.name for f in fm.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in have:
            return name
    print("[提示] 没有找到中文字体，图中中文可能显示为方框。可设置环境变量 FIG_FONT=<字体文件路径>。")
    return "DejaVu Sans"

plt.rcParams.update({
    "font.family": [_pick_font(), "DejaVu Sans"],
    "font.size": FS, "axes.titlesize": FS + 1, "axes.titleweight": "bold", "axes.labelsize": FS,
    "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1, "legend.fontsize": FS - 1,
    "axes.edgecolor": "#aab2bd", "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "legend.frameon": False, "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "lines.linewidth": 2.6, "lines.solid_capstyle": "round",
})


def fig(w, h, **kw):
    return plt.figure(figsize=(w / 100.0, h / 100.0), **kw)

def save(f, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, name + ".png")
    f.savefig(out, dpi=200)
    plt.close(f)
    return out

def title(ax, text, **kw):
    kw.setdefault("loc", "left"); kw.setdefault("pad", 10)
    ax.set_title(text, **kw)

def ygrid(ax):
    ax.grid(axis="y", color=GRID, linewidth=1.0); ax.set_axisbelow(True)

def xgrid(ax):
    ax.grid(axis="x", color=GRID, linewidth=1.0); ax.set_axisbelow(True)
