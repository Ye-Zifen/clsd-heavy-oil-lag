"""
特征工程：把井组, 日期 t, 采出指标映射成一行数值特征。
"""
from __future__ import annotations

import itertools
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from scipy.stats import t as tdist

PROD_INDICATORS = [
    "LIQ_PROD_DAILY",
    "OIL_PROD_DAILY",
    "WATER_CUT",
    "WH_TEMP",
    "MAX_TUBING_PRES",
]
MAX_LAG = 45

XCH = ("ffill", "fill0", "bin", "dabs")
YCH = ("ffill", "dabs")
WINS = (("t21", 21), ("t45", 45), ("t90", 90), ("t150", 150))
STATS = ("abs", "cos")
RAW_CURVE = (("ffill", "ffill", "t90", "abs"),)
VAR_EPS = 1e-10


def _read(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def load_data(data_dir: str, include_test: bool) -> dict:
    tr_dir = os.path.join(data_dir, "train")
    te_dir = os.path.join(data_dir, "test")
    gi = _read(os.path.join(tr_dir, "well_group_info.csv"))
    lag_tr = _read(os.path.join(tr_dir, "optimal_lag_days.csv"))
    inj = [_read(os.path.join(tr_dir, "well_inj_data.csv"))]
    prod = [_read(os.path.join(tr_dir, "well_prod_data.csv"))]
    if include_test:
        inj.append(_read(os.path.join(te_dir, "well_inj_data.csv")))
        prod.append(_read(os.path.join(te_dir, "well_prod_data.csv")))
        gi_te = os.path.join(te_dir, "well_group_info.csv")
        if os.path.exists(gi_te):
            gi = (pd.concat([gi, _read(gi_te)], ignore_index=True)
                  .drop_duplicates(["WELL_GROUP_NAME", "WELL_NAME", "WELL_PURPOSE"]).reset_index(drop=True))
    inj = pd.concat(inj, ignore_index=True)
    prod = pd.concat(prod, ignore_index=True)
    for df in (lag_tr, inj, prod):
        df["PROD_DATE"] = pd.to_datetime(df["PROD_DATE"])
    lag_tr = lag_tr.drop_duplicates(
        ["WELL_GROUP_NAME", "PROD_DATE", "INJ_INDICATOR", "PROD_INDICATOR"]).reset_index(drop=True)
    return {"gi": gi, "lag_train": lag_tr, "inj": inj, "prod": prod}


def _cs(a):
    c = np.zeros(len(a) + 1, dtype=float)
    c[1:] = np.cumsum(a)
    return c


class SeriesBank:

    def __init__(self, raw: dict, extra_dates=None):
        gi, inj, prod = raw["gi"], raw["inj"], raw["prod"]
        lo = min(inj.PROD_DATE.min(), prod.PROD_DATE.min())
        hi = max(inj.PROD_DATE.max(), prod.PROD_DATE.max())
        if extra_dates is not None and len(extra_dates):
            lo = min(lo, pd.Timestamp(min(extra_dates)))
            hi = max(hi, pd.Timestamp(max(extra_dates)))
        self.index = pd.date_range(lo, hi, freq="D")
        self.pos = {d: i for i, d in enumerate(self.index)}
        self.n = len(self.index)

        declared = gi[gi.WELL_PURPOSE == "生产井"].groupby("WELL_GROUP_NAME").WELL_NAME.apply(set).to_dict()
        self.INJ = {}
        for g, s in inj.groupby("WELL_GROUP_NAME"):
            s = s[~s.WELL_NAME.isin(declared.get(g, set()))]
            if len(s) == 0:
                continue
            self.INJ[g] = (s.groupby("PROD_DATE").INJ_VOL_DAILY.sum(min_count=1)
                           .reindex(self.index).to_numpy(dtype=float))

        dual = set(gi[gi.WELL_PURPOSE == "注汽井"].WELL_NAME)
        self.PRO = {}
        for g, s in prod.groupby("WELL_GROUP_NAME"):

            pw = declared.get(g, set())
            pw2 = pw - dual
            s2 = s[s.WELL_NAME.isin(pw2 if pw2 else pw)]
            if len(s2) == 0:
                s2 = s
            self.PRO[g] = {ind: s2.groupby("PROD_DATE")[ind].mean()
                           .reindex(self.index).to_numpy(dtype=float) for ind in PROD_INDICATORS}
        self._cache = {}

    def _xch(self, g):
        x = self.INJ.get(g)
        if x is None:
            return None
        x0 = np.nan_to_num(x, nan=0.0)
        xf = pd.Series(x).ffill().fillna(0.0).to_numpy()
        return {"ffill": xf, "fill0": x0, "bin": (x0 > 0).astype(float),
                "dabs": np.abs(np.diff(xf, prepend=xf[0]))}

    def _ych(self, g, ind):
        v = self.PRO.get(g, {}).get(ind)
        if v is None:
            return None
        vf = pd.Series(v).ffill().fillna(0.0).to_numpy()   # 只向前填充，首个观测之前记 0
        return {"ffill": vf, "dabs": np.abs(np.diff(vf, prepend=vf[0]))}

    def recipes(self, g, ind):
        key = (g, ind)
        if key in self._cache:
            return self._cache[key]
        X, Y = self._xch(g), self._ych(g, ind)
        out = {}
        if X is None or Y is None:
            self._cache[key] = out
            return out
        n = self.n
        t = np.arange(n)
        for xn, yn in itertools.product(XCH, YCH):
            xv, yv = X[xn], Y[yn]
            C = {}
            for L in range(MAX_LAG + 1):
                a = np.zeros(n)
                if L == 0:
                    a[:] = xv
                else:
                    a[L:] = xv[:-L]
                C[L] = (_cs(a), _cs(yv), _cs(a * a), _cs(yv * yv), _cs(a * yv))
            for wn, W in WINS:
                lo, hi = np.maximum(t - W + 1, 0), t + 1      # 尾随窗 [t-W+1, t]
                cw = (hi - lo).astype(float)
                for st in STATS:
                    M = np.full((MAX_LAG + 1, n), np.nan)
                    for L in range(MAX_LAG + 1):
                        Sa, Sb, Saa, Sbb, Sab = C[L]
                        sa = Sa[hi] - Sa[lo]; sb = Sb[hi] - Sb[lo]
                        saa = Saa[hi] - Saa[lo]; sbb = Sbb[hi] - Sbb[lo]; sab = Sab[hi] - Sab[lo]
                        if st == "abs":
                            cov = sab / cw - (sa / cw) * (sb / cw)
                            va = saa / cw - (sa / cw) ** 2
                            vb = sbb / cw - (sb / cw) ** 2
                            c = np.abs(cov / np.sqrt(va * vb))
                            # 序列是台阶函数，窗口内方差常常精确为 0，此时相关性无定义。
                            # 不加这道闸，cov/0 会溢出成 inf，而 HGB 会把 inf 分到最高分箱，
                            # 等于把"无定义"误当成"相关性极强"（此前 1.1% 的特征元素、63% 的样本行中招）。
                            c = np.where((va > VAR_EPS) & (vb > VAR_EPS), c, np.nan)
                        else:
                            c = np.abs(sab / np.sqrt(saa * sbb))
                            c = np.where((saa > VAR_EPS) & (sbb > VAR_EPS), c, np.nan)
                        c = np.where(np.isfinite(c), np.minimum(c, 1.0), np.nan)
                        c[cw < 6] = np.nan
                        M[L] = c
                    A = np.where(np.isnan(M), -1.0, M)
                    valid = ~np.isnan(M[0])
                    pre = "%s.%s.%s.%s" % (xn, yn, wn, st)
                    smax = A[0:4].max(axis=0); lmax = A[4:].max(axis=0)
                    out[pre + ".sml"] = np.where(valid, smax - lmax, np.nan)
                    out[pre + ".ama"] = np.where(valid, A.argmax(axis=0).astype(float), np.nan)
                    out[pre + ".mxa"] = np.where(valid, A.max(axis=0), np.nan)
                    out[pre + ".amas"] = np.where(valid, A[0:4].argmax(axis=0).astype(float), np.nan)
                    out[pre + ".amal"] = np.where(valid, (A[4:].argmax(axis=0) + 4).astype(float), np.nan)
                    out[pre + ".smax"] = np.where(valid, smax, np.nan)
                    if (xn, yn, wn, st) in RAW_CURVE:
                        for L in range(MAX_LAG + 1):
                            out["%s.L%02d" % (pre, L)] = M[L]
        self._cache[key] = out
        return out


def recipe_names():
    names = []
    for xn, yn in itertools.product(XCH, YCH):
        for wn, _ in WINS:
            for st in STATS:
                pre = "%s.%s.%s.%s" % (xn, yn, wn, st)
                names += [pre + s for s in (".sml", ".ama", ".mxa", ".amas", ".amal", ".smax")]
                if (xn, yn, wn, st) in RAW_CURVE:
                    names += ["%s.L%02d" % (pre, L) for L in range(MAX_LAG + 1)]
    return sorted(names)


def build_frame(bank: SeriesBank, keys: pd.DataFrame) -> pd.DataFrame:
    keys = keys.reset_index(drop=True)
    ti = keys.PROD_DATE.map(bank.pos)
    ok = ti.notna().to_numpy()
    ti = ti.fillna(0).astype(int).to_numpy()
    probe = recipe_names()
    cols = {nm: np.full(len(keys), np.nan) for nm in probe}
    for (g, ind), idx in keys.groupby(["WELL_GROUP_NAME", "PROD_INDICATOR"]).indices.items():
        f = bank.recipes(g, ind)
        if not f:
            continue
        tt = ti[idx]
        for nm in probe:
            if nm in f:
                cols[nm][idx] = f[nm][tt]
    F = pd.DataFrame(cols)
    F["ind_code"] = keys.PROD_INDICATOR.map({v: i for i, v in enumerate(PROD_INDICATORS)}).astype(float)
    F.loc[~ok, :] = np.nan
    return F

ML = 45
SUMS = ("sml", "ama", "mxa", "amas", "amal", "smax", "mean", "q2")


def _rank_rows(M):
    R = rankdata(M, axis=1)
    return R - R.mean(axis=1, keepdims=True)


def _corr_ranked(Rx, Ry):
    vx = (Rx * Rx).sum(1)
    vy = (Ry * Ry).sum(1)
    c = (Rx * Ry).sum(1) / np.sqrt(vx * vy)
    c[(vx < 1e-9) | (vy < 1e-9)] = np.nan
    return c


def trail_curves(x, y, tt, W):
    idx = tt[:, None] - np.arange(W)[::-1][None, :]
    okY = (idx >= 0).all(1)
    Ry = _rank_rows(y[np.clip(idx, 0, None)])
    C = np.full((ML + 1, len(tt)), np.nan)
    for L in range(ML + 1):
        ix = idx - L
        ok = okY & (ix >= 0).all(1)
        c = np.abs(_corr_ranked(_rank_rows(x[np.clip(ix, 0, None)]), Ry))
        c[~ok] = np.nan
        C[L] = c
    return C


def fixed_curves(x, y, tt, W):
    idx = tt[:, None] - np.arange(W)[::-1][None, :]
    ok = (idx >= 0).all(1)
    idx = np.clip(idx, 0, None)
    Xw, Yw = x[idx], y[idx]
    C = np.full((ML + 1, len(tt)), np.nan)
    for L in range(ML + 1):
        c = np.abs(_corr_ranked(_rank_rows(Xw[:, :W - L]), _rank_rows(Yw[:, L:])))
        c[~ok] = np.nan
        C[L] = c
    return C


def pval_curves(C, W):
    n = (W - np.arange(ML + 1)).astype(float)[:, None]
    r = np.clip(C, 0.0, 1.0 - 1e-12)
    ts = r * np.sqrt((n - 2.0) / (1.0 - r * r))
    p = 2.0 * tdist.sf(ts, n - 2.0)
    out = -np.log10(np.maximum(p, 1e-300))
    out[np.isnan(C)] = np.nan
    return out


def _summ(C, pre, out, j, full=False):
    A = np.where(np.isnan(C), -1.0, C)
    valid = ~np.isnan(C).all(0)
    sm = A[:4].max(0)
    lm = A[4:].max(0)
    out[pre + ".sml"][j] = np.where(valid, sm - lm, np.nan)
    out[pre + ".ama"][j] = np.where(valid, A.argmax(0), np.nan)
    out[pre + ".mxa"][j] = np.where(valid, A.max(0), np.nan)
    out[pre + ".amas"][j] = np.where(valid, A[:4].argmax(0), np.nan)
    out[pre + ".amal"][j] = np.where(valid, A[4:].argmax(0) + 4, np.nan)
    out[pre + ".smax"][j] = np.where(valid, sm, np.nan)
    with np.errstate(all="ignore"):
        out[pre + ".mean"][j] = np.where(valid, np.nanmean(np.where(np.isnan(C), np.nan, C), 0), np.nan)
    srt = np.sort(A, 0)
    out[pre + ".q2"][j] = np.where(valid, srt[-1] - srt[-2], np.nan)
    if full:
        for L in range(ML + 1):
            out["%s.L%02d" % (pre, L)][j] = C[L]

LIN_FIXED = [(75, 0), (90, 0), (105, 0), (120, 0), (90, 3), (75, 3), (105, 3)]
LIN_FIXED_FULL = {(90, 0), (90, 3)}
LIN_TRAIL = [60, 90, 120]
RULE_KS = (0.5, 1.0, 1.5, 2.0)
RULE_SIG = (4, 6, 8)
USE_SHAPE_GRID = False


def _lin(y):
    return pd.Series(y).interpolate(limit_direction="both").to_numpy()


def _ma(v, k):
    return pd.Series(v).rolling(k, min_periods=1).mean().to_numpy() if k > 1 else v


def kendall_fixed(x, y, tt, W, chunk=150):
    n_all = len(tt)
    TAU = np.full((ML + 1, n_all), np.nan)
    NLP = np.full((ML + 1, n_all), np.nan)
    for s0 in range(0, n_all, chunk):
        sl = slice(s0, s0 + chunk)
        t = tt[sl]
        idx = t[:, None] - np.arange(W)[::-1][None, :]
        ok = (idx >= 0).all(1)
        idx = np.clip(idx, 0, None)
        Xw, Yw = x[idx], y[idx]
        for L in range(ML + 1):
            n = W - L
            a = Xw[:, :n]; b = Yw[:, L:]
            sa = np.sign(a[:, :, None] - a[:, None, :])
            sb = np.sign(b[:, :, None] - b[:, None, :])
            S = (sa * sb).sum((1, 2)) / 2.0
            n0 = n * (n - 1) / 2.0
            ca = (sa == 0).sum(2).astype(float)
            cb = (sb == 0).sum(2).astype(float)
            n1 = (ca - 1).sum(1) / 2.0; n2 = (cb - 1).sum(1) / 2.0
            vt = ((ca - 1) * (2 * ca + 5)).sum(1); vu = ((cb - 1) * (2 * cb + 5)).sum(1)
            wt = ((ca - 1) * (ca - 2)).sum(1); wu = ((cb - 1) * (cb - 2)).sum(1)
            st = (ca - 1).sum(1); su = (cb - 1).sum(1)
            den = np.sqrt((n0 - n1) * (n0 - n2))
            v0 = n * (n - 1) * (2 * n + 5)
            var = (v0 - vt - vu) / 18.0 + (st * su) / (2.0 * n * (n - 1)) + (wt * wu) / (9.0 * n * (n - 1) * (n - 2))
            with np.errstate(all="ignore"):
                tau = np.abs(S / den)
                z = S / np.sqrt(var)
                nlp = -np.log10(np.maximum(2 * norm.sf(np.abs(z)), 1e-300))
            bad = (den <= 0) | (var <= 0) | ~ok
            tau[bad] = np.nan; nlp[bad] = np.nan
            TAU[L, sl] = tau; NLP[L, sl] = nlp
    return TAU, NLP


def rule_feats(LP, pre, out, j):
    A = np.where(np.isnan(LP), -np.inf, LP).T
    valid = np.isfinite(A).any(1)
    mx = A.max(1, keepdims=True)
    for k in RULE_KS:
        f = (A >= mx - k).argmax(1).astype(float); f[~valid] = np.nan
        out["%s.first%.1f" % (pre, k)][j] = f
    for a in RULE_SIG:
        s = A >= a
        f = np.where(s.any(1), s.argmax(1), A.argmax(1)).astype(float); f[~valid] = np.nan
        out["%s.sig%d" % (pre, a)][j] = f
    with np.errstate(all="ignore"):
        out[pre + ".nwithin1"][j] = np.where(valid, (A >= mx - 1).sum(1), np.nan)
        out[pre + ".shortgap"][j] = np.where(valid, A[:, :4].max(1) - mx[:, 0], np.nan)


def _rule_names(pre):
    return (["%s.first%.1f" % (pre, k) for k in RULE_KS] + ["%s.sig%d" % (pre, a) for a in RULE_SIG]
            + [pre + ".nwithin1", pre + ".shortgap"])


def feature_names():
    names = []
    for W, ma in LIN_FIXED:
        for fam in ("fl", "pl"):
            pre = "%s.%d.ma%d" % (fam, W, ma)
            names += [pre + "." + s for s in SUMS]
        if (W, ma) in LIN_FIXED_FULL:
            names += ["pl.%d.ma%d.L%02d" % (W, ma, L) for L in range(ML + 1)]
            names += _rule_names("rl.%d.ma%d" % (W, ma))
    for W in LIN_TRAIL:
        pre = "tl.%d" % W
        names += [pre + "." + s for s in SUMS]
        if W == 90:
            names += ["%s.L%02d" % (pre, L) for L in range(ML + 1)]
    names += ["kl.90." + s for s in SUMS] + ["kl.90.L%02d" % L for L in range(ML + 1)] + _rule_names("rkl.90")
    return names


GAP_RAMP_SLOPE = 0.02


def _causal_groups(yr, y0, tt):
    obs = ~np.isnan(yr)
    last = np.where(obs, np.arange(len(yr)), -1)
    last = np.maximum.accumulate(last)
    lo = last[tt]
    groups = []
    clean = obs[tt]
    if clean.any():
        groups.append((y0, np.where(clean)[0]))
    for v in np.unique(lo[(~clean) & (lo >= 0)]):
        ym = y0.copy()
        w = yr[max(0, v - 89):v + 1]
        w = w[~np.isnan(w)]
        sd = float(np.std(w)) if len(w) else 0.0
        step = GAP_RAMP_SLOPE * sd if sd > 0 else 1e-9 * max(1.0, abs(float(yr[v])))
        ym[v + 1:] = yr[v] - step * np.arange(1, len(ym) - v)
        groups.append((ym, np.where((~clean) & (lo == v))[0]))
    return groups


def build_lin_frame(INJ: dict, PRO: dict, keys_g, keys_i, keys_t) -> pd.DataFrame:
    keys_t = np.asarray(keys_t, dtype=int)
    n = len(keys_t)
    names = feature_names()
    out = {nm: np.full(n, np.nan, dtype=np.float64) for nm in names}
    df = pd.DataFrame({"g": np.asarray(keys_g), "i": np.asarray(keys_i), "t": keys_t, "j": np.arange(n)})
    for (g, ind), d in df.groupby(["g", "i"], sort=True):
        jj = d.j.to_numpy(); tt_all = d.t.to_numpy()
        xr = INJ.get(g); yr = PRO.get(g, {}).get(ind)
        if xr is None or yr is None or np.isnan(yr).all():
            continue
        x0 = np.nan_to_num(xr)
        y_full = _lin(yr)
        for y0, sel in _causal_groups(yr, y_full, tt_all):
            j = jj[sel]; tt = tt_all[sel]
            for W, ma in LIN_FIXED:
                C = fixed_curves(_ma(x0, ma), _ma(y0, ma), tt, W)
                _summ(C, "fl.%d.ma%d" % (W, ma), out, j)
                P = pval_curves(C, W)
                _summ(P, "pl.%d.ma%d" % (W, ma), out, j, full=(W, ma) in LIN_FIXED_FULL)
                if (W, ma) in LIN_FIXED_FULL:
                    rule_feats(P, "rl.%d.ma%d" % (W, ma), out, j)
            for W in LIN_TRAIL:
                C = trail_curves(x0, y0, tt, W)
                _summ(C, "tl.%d" % W, out, j, full=(W == 90))
            TAU, NLP = kendall_fixed(x0, y0, tt, 90)
            _summ(NLP, "kl.90", out, j, full=True)
            rule_feats(NLP, "rkl.90", out, j)
    return pd.DataFrame(out).astype(np.float32)

def _build_block(bank, keys):
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    keys = keys.reset_index(drop=True)
    F = build_frame(bank, keys).astype(np.float32)
    ti = keys.PROD_DATE.map(bank.pos).fillna(-1).astype(int).to_numpy()
    R = build_lin_frame(bank.INJ, bank.PRO, keys.WELL_GROUP_NAME.to_numpy(),
                        keys.PROD_INDICATOR.to_numpy(), ti)
    R.loc[ti < 0, :] = np.nan
    S = shape_features(R)
    T = window_stats(bank, keys)
    parts = [F.reset_index(drop=True), R.reset_index(drop=True), S, T]
    if USE_SHAPE_GRID:
        parts.append(shape_grid_features(R))
    return pd.concat(parts, axis=1), (F.shape[1], R.shape[1], S.shape[1] + (25 if USE_SHAPE_GRID else 0), T.shape[1])


def _build_parallel(bank, keys, n_jobs):
    from joblib import Parallel, delayed
    sizes = keys.groupby("WELL_GROUP_NAME", sort=True).size().sort_values(ascending=False, kind="stable")
    n_bins = max(1, min(len(sizes), n_jobs * 4))
    bins = [[] for _ in range(n_bins)]
    load = [0] * n_bins
    for g, c in sizes.items():
        k = load.index(min(load))
        bins[k].append(g)
        load[k] += int(c)
    gcol = keys.WELL_GROUP_NAME.to_numpy()
    idxs = [np.where(np.isin(gcol, b))[0] for b in bins if b]
    bank._cache = {}
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_build_block)(bank, keys.iloc[ix]) for ix in idxs)
    X = pd.concat([r[0].set_axis(ix, axis=0) for r, ix in zip(res, idxs)], axis=0).sort_index(kind="stable")
    return X.reset_index(drop=True), res[0][1]


def build_features(bank, keys, verbose=True, tag="", n_jobs=1):
    keys = keys.reset_index(drop=True)
    X = None
    if n_jobs and n_jobs > 1 and len(keys) >= 2000:
        try:
            X, dims = _build_parallel(bank, keys, int(n_jobs))
        except Exception as e:
            print("多进程构造特征失败（%s: %s），改为单进程" % (type(e).__name__, e), flush=True)
            X = None
    if X is None:
        X, dims = _build_block(bank, keys)
    if verbose:
        print("  %s特征：互相关配方库 %d 维 + 插值秩相关族 %d 维 + 曲线形状 %d 维 + 窗口统计 %d 维 共 %d 维"
              % (tag, dims[0], dims[1], dims[2], dims[3], X.shape[1]), flush=True)
    return X

SHAPE_FAMS = ("pl.90.ma0", "pl.90.ma3", "kl.90", "tl.90")
SHAPE_GRID = np.array([0, 1, 2, 3, 6] + list(range(9, 46)))


def shape_features(R: pd.DataFrame) -> pd.DataFrame:
    out = {}
    with np.errstate(all="ignore"):
        for fam in SHAPE_FAMS:
            c = np.stack([R["%s.L%02d" % (fam, k)].to_numpy(np.float64) for k in range(ML + 1)], 1)
            mx = np.nanmax(np.where(np.isnan(c), -np.inf, c), 1)
            mxs = np.where(mx > 0, mx, np.nan)
            out[fam + ".d30"] = c[:, 3] - c[:, 0]
            out[fam + ".d10"] = c[:, 1] - c[:, 0]
            out[fam + ".d21"] = c[:, 2] - c[:, 1]
            out[fam + ".d32"] = c[:, 3] - c[:, 2]
            out[fam + ".rise48"] = np.nanmax(c[:, 4:9], 1) - c[:, 3]
            out[fam + ".rise912"] = np.nanmax(c[:, 9:13], 1) - np.nanmax(c[:, 4:9], 1)
            out[fam + ".max03_rel"] = np.nanmax(c[:, :4], 1) / mxs
            out[fam + ".c3_rel"] = c[:, 3] / mxs
            out[fam + ".c0_rel"] = c[:, 0] / mxs
    return pd.DataFrame(out).astype(np.float32)


def shape_grid_features(R: pd.DataFrame) -> pd.DataFrame:
    if True:
        out2 = {}
        with np.errstate(all="ignore"):
            for fam in SHAPE_FAMS:
                c = np.stack([R["%s.L%02d" % (fam, k)].to_numpy(np.float64) for k in range(ML + 1)], 1)
                A = np.where(np.isnan(c), -np.inf, c)
                mx = A.max(1)
                valid = np.isfinite(mx)
                mxs = np.where(mx > 0, mx, np.nan)
                out2[fam + ".c6_rel"] = c[:, 6] / mxs
                out2[fam + ".c9_rel"] = c[:, 9] / mxs
                sub = c[:, :13]
                lm = (sub[:, 1:-1] >= sub[:, :-2]) & (sub[:, 1:-1] >= sub[:, 2:])
                out2[fam + ".nlocmax012"] = lm.sum(1).astype(float)
                fl = np.where(lm.any(1), lm.argmax(1) + 1, np.nan)
                out2[fam + ".firstlocmax"] = np.where(sub[:, 0] >= sub[:, 1], 0, fl)
                if fam.startswith("pl") or fam.startswith("kl"):
                    Ag = A[:, SHAPE_GRID]
                    mg = Ag.max(1, keepdims=True)
                    for f in (0.6, 0.7, 0.8):
                        r = SHAPE_GRID[(Ag >= f * mg).argmax(1)].astype(float)
                        r[~valid] = np.nan
                        out2[fam + ".gff%.1f" % f] = r
    return pd.DataFrame(out2).astype(np.float32)

_ST1 = ['st.inj_nz', 'st.inj_obs', 'st.inj_chg', 'st.inj_since_chg', 'st.inj_since_nz', 'st.inj_cv', 'st.inj_maxmean',
        'st.prod_obs', 'st.prod_gap', 'st.prod_cv', 'st.prod_ac1', 'st.prod_uniq', 'st.prod_trend', 'st.prod_chg']
_ST2 = ['inj_nz', 'inj_obs', 'inj_chg', 'inj_since_chg', 'inj_first_chg', 'inj_since_nz', 'inj_cv', 'inj_maxmean',
        'inj_uniq', 'inj_bigjump', 'inj_bigjump_pos', 'inj_recent_ratio',
        'prod_obs', 'prod_cv', 'prod_ac1', 'prod_ac7', 'prod_uniq', 'prod_trend', 'prod_chg', 'prod_bigjump',
        'prod_recent_z', 'prod_ties']
ST_WINS = (30, 60, 90, 120)

def _st1(xr, yr, t, out, j, W=90):
    if t < W - 1:
        return
    s = t - W + 1
    if xr is not None:
        xw = xr[s:t + 1]; x0 = np.nan_to_num(xw)
        out['st.inj_nz'][j] = (x0 > 0).mean(); out['st.inj_obs'][j] = (~np.isnan(xw)).mean()
        ch = np.where(np.diff(x0) != 0)[0]; out['st.inj_chg'][j] = len(ch)
        out['st.inj_since_chg'][j] = (W - 2 - ch[-1]) if len(ch) else W
        nz = np.where(x0 > 0)[0]; out['st.inj_since_nz'][j] = (W - 1 - nz[-1]) if len(nz) else W
        m = x0.mean(); out['st.inj_cv'][j] = x0.std() / m if m > 0 else np.nan
        out['st.inj_maxmean'][j] = x0.max() / m if m > 0 else np.nan
    if yr is not None:
        yw = yr[s:t + 1]; ob = ~np.isnan(yw); out['st.prod_obs'][j] = ob.mean()
        past = np.where(~np.isnan(yr[:t + 1]))[0]; out['st.prod_gap'][j] = (t - past[-1]) if len(past) else np.nan
        v = yw[ob]
        if len(v) >= 5:
            m = np.abs(v.mean()); out['st.prod_cv'][j] = v.std() / m if m > 0 else np.nan
            a = v[:-1] - v[:-1].mean(); b = v[1:] - v[1:].mean(); den = np.sqrt((a * a).sum() * (b * b).sum())
            out['st.prod_ac1'][j] = (a * b).sum() / den if den > 0 else np.nan
            out['st.prod_uniq'][j] = len(np.unique(v)) / len(v)
            tt = np.where(ob)[0]; out['st.prod_trend'][j] = np.corrcoef(tt, v)[0, 1] if v.std() > 0 else np.nan
            out['st.prod_chg'][j] = (np.diff(v) != 0).mean()

def _st2(xr, yr, t, W, pre, out, j):
    if t < W - 1:
        return
    s = t - W + 1
    if xr is not None:
        xw = xr[s:t + 1]; x0 = np.nan_to_num(xw)
        out[pre + 'inj_nz'][j] = (x0 > 0).mean(); out[pre + 'inj_obs'][j] = (~np.isnan(xw)).mean()
        ch = np.where(np.diff(x0) != 0)[0]; out[pre + 'inj_chg'][j] = len(ch)
        out[pre + 'inj_since_chg'][j] = (W - 2 - ch[-1]) if len(ch) else W
        out[pre + 'inj_first_chg'][j] = (W - 2 - ch[0]) if len(ch) else W
        nz = np.where(x0 > 0)[0]; out[pre + 'inj_since_nz'][j] = (W - 1 - nz[-1]) if len(nz) else W
        m = x0.mean(); out[pre + 'inj_cv'][j] = x0.std() / m if m > 0 else np.nan
        out[pre + 'inj_maxmean'][j] = x0.max() / m if m > 0 else np.nan
        out[pre + 'inj_uniq'][j] = len(np.unique(x0))
        mx = x0.max(); out[pre + 'inj_bigjump'][j] = (np.abs(np.diff(x0)) > 0.5 * mx).sum() if mx > 0 else 0
        dj = np.abs(np.diff(x0)); out[pre + 'inj_bigjump_pos'][j] = (W - 2 - dj.argmax()) if mx > 0 else np.nan
        out[pre + 'inj_recent_ratio'][j] = (x0[-15:].mean() / m) if m > 0 else np.nan
    if yr is not None:
        yw = yr[s:t + 1]; ob = ~np.isnan(yw); out[pre + 'prod_obs'][j] = ob.mean()
        v = yw[ob]
        if len(v) >= 5:
            m = np.abs(v.mean()); out[pre + 'prod_cv'][j] = v.std() / m if m > 0 else np.nan
            a = v[:-1] - v[:-1].mean(); b = v[1:] - v[1:].mean(); den = np.sqrt((a * a).sum() * (b * b).sum())
            out[pre + 'prod_ac1'][j] = (a * b).sum() / den if den > 0 else np.nan
            if len(v) > 8:
                a = v[:-7] - v[:-7].mean(); b = v[7:] - v[7:].mean(); den = np.sqrt((a * a).sum() * (b * b).sum())
                out[pre + 'prod_ac7'][j] = (a * b).sum() / den if den > 0 else np.nan
            out[pre + 'prod_uniq'][j] = len(np.unique(v)) / len(v)
            tt = np.where(ob)[0]; out[pre + 'prod_trend'][j] = np.corrcoef(tt, v)[0, 1] if v.std() > 0 else np.nan
            out[pre + 'prod_chg'][j] = (np.diff(v) != 0).mean()
            sd = v.std(); out[pre + 'prod_bigjump'][j] = (np.abs(np.diff(v)) > 2 * sd).sum() if sd > 0 else 0
            out[pre + 'prod_recent_z'][j] = (v[-7:].mean() - v.mean()) / sd if sd > 0 else 0
            out[pre + 'prod_ties'][j] = 1 - len(np.unique(v)) / len(v)


def window_stats(bank: SeriesBank, keys: pd.DataFrame) -> pd.DataFrame:
    keys = keys.reset_index(drop=True)
    n = len(keys)
    ti = keys.PROD_DATE.map(bank.pos).fillna(-1).astype(int).to_numpy()
    o1 = {k: np.full(n, np.nan) for k in _ST1}
    o2 = {}
    for W in ST_WINS:
        for nm in _ST2:
            o2['st%d.%s' % (W, nm)] = np.full(n, np.nan)
    with np.errstate(all="ignore"):
        for (g, ind), d in keys.assign(t=ti).groupby(['WELL_GROUP_NAME', 'PROD_INDICATOR']):
            xr = bank.INJ.get(g); yr = bank.PRO.get(g, {}).get(ind)
            for j, t in zip(d.index, d.t):
                if t < 0:
                    continue
                _st1(xr, yr, t, o1, j)
                for W in ST_WINS:
                    _st2(xr, yr, t, W, 'st%d.' % W, o2, j)
    return pd.concat([pd.DataFrame(o1), pd.DataFrame(o2)], axis=1).astype(np.float32)
