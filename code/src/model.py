"""
训练、模型存取与逐样本推理。
"""
from __future__ import annotations

import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from .feature import load_data, SeriesBank, build_features

CAND = np.arange(46)
SHORT_VALS = np.arange(4)


# ---------------- 训练 ----------------
def _curve_cols(cfg):
    pre = cfg["history"]["similarity"]["curve_prefix"]
    return ["%s%02d" % (pre, i) for i in range(46)]


def _curve_corr(A, B):
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    return (A * B).sum(axis=1) / np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1) + 1e-12)


def _et_matrix(X, cfg):
    return X.fillna(cfg["segment_classifier"]["missing_value"]).to_numpy(np.float32)

def _seg_targets(y, kind):
    if kind == "binary":
        return (y <= 3).astype(int)
    if kind == "c7":
        return np.digitize(y, [1, 2, 3, 4, 10, 26]).astype(int)
    raise ValueError(kind)


def _p_short(et, Xe):
    P = et.predict_proba(Xe)
    if getattr(et, "seg_kind", "binary") == "binary":
        return P[:, list(et.classes_).index(1)]
    return P[:, et.classes_ <= 3].sum(axis=1)


def n_jobs_of(cfg):
    nj = int((cfg.get("runtime") or {}).get("n_jobs", 1) or 0)
    if nj <= 0:
        nj = min(os.cpu_count() or 1, 8)
    return max(1, nj)


def _fit_ets(Xe, y, w, cfg):
    sc = {k: v for k, v in cfg["segment_classifier"].items() if k not in ("missing_value", "random_states", "targets")}
    kinds = cfg["segment_classifier"].get("targets", ["binary"] * len(cfg["segment_classifier"]["random_states"]))
    ets = []
    for rs, kind in zip(cfg["segment_classifier"]["random_states"], kinds):
        et = ExtraTreesClassifier(n_jobs=n_jobs_of(cfg), random_state=rs, **sc)
        et.fit(Xe, _seg_targets(y, kind), sample_weight=w)
        et.n_jobs = 1
        et.seg_kind = kind
        ets.append(et)
    return ets


def _anchor_model(lag, Xe, y, anchor, cfg):
    hd = cfg["history"]["anchor_holdout_days"]
    cutB = anchor - pd.Timedelta(days=hd)
    trB = (lag.PROD_DATE <= cutB).to_numpy()
    teB = ~trB
    wB = 0.5 ** ((cutB - lag.PROD_DATE).dt.days.to_numpy() / cfg["training"]["recency_halflife_days"])
    etsB = _fit_ets(Xe[trB], y[trB], wB[trB], cfg)
    pB = np.mean([_p_short(e, Xe[teB]) for e in etsB], axis=0)
    del etsB
    lb = lag.loc[teB, ["WELL_GROUP_NAME", "PROD_INDICATOR", "PROD_DATE"]]
    anchor_pet = {(g, i, pd.Timestamp(d)): float(v) for g, i, d, v in zip(lb.WELL_GROUP_NAME, lb.PROD_INDICATOR, lb.PROD_DATE, pB)}
    return anchor_pet

def train(data_dir, cfg, verbose=True):
    t0 = time.time()
    raw = load_data(data_dir, include_test=False)
    lag = raw["lag_train"]
    if verbose:
        print("[训练] 只用 train/ 目录构造训练特征 (%d 行，并行度 %d)" % (len(lag), n_jobs_of(cfg)), flush=True)
    X = build_features(SeriesBank(raw), lag[["WELL_GROUP_NAME", "PROD_DATE", "PROD_INDICATOR"]],
                       verbose=verbose, tag="训练", n_jobs=n_jobs_of(cfg))
    if verbose:
        print("[训练] 特征构造用时 %.0f 秒" % (time.time() - t0), flush=True)
    t1 = time.time()
    y = lag.OPTIMAL_LAG_DAYS.to_numpy()
    anchor = lag.PROD_DATE.max()
    w = 0.5 ** ((anchor - lag.PROD_DATE).dt.days.to_numpy() / cfg["training"]["recency_halflife_days"])

    Xe = _et_matrix(X, cfg)
    ets = _fit_ets(Xe, y, w, cfg)

    anchor_pet = {}
    if cfg["history"]["enabled"]:
        anchor_pet = _anchor_model(lag, Xe, y, anchor, cfg)
    del Xe

    anchor_curves = None
    if cfg["history"]["enabled"] and cfg["history"].get("similarity", {}).get("enabled", False):
        cc = _curve_cols(cfg)
        anchor_curves = {"keys": lag[["WELL_GROUP_NAME", "PROD_INDICATOR", "PROD_DATE"]].reset_index(drop=True),
                         "curves": np.nan_to_num(X[cc].to_numpy(np.float64)).astype(np.float32)}

    hv = cfg["training"].get("value_halflife_days", cfg["training"]["recency_halflife_days"])
    w = 0.5 ** ((anchor - lag.PROD_DATE).dt.days.to_numpy() / hv)
    fix = cfg["lightgbm_common"]
    ms = y <= 3
    clf_s = lgb.LGBMClassifier(**fix, **cfg["short_value_model"])
    clf_s.fit(X[ms], y[ms], sample_weight=w[ms])
    ml = y >= 4
    reg = lgb.LGBMRegressor(**fix, **cfg["long_value_model"])
    reg.fit(X[ml], y[ml], sample_weight=w[ml])
    if verbose:
        print("[训练] 完成：ExtraTrees×%d 分段分类 + LightGBM 短段取值 4 分类 + LightGBM 长段回归（模型训练用时 %.0f 秒）"
              % (len(ets), time.time() - t1), flush=True)
    return {"columns": list(X.columns), "segment": ets, "short_value": clf_s, "long_value": reg,
            "anchor_pet": anchor_pet, "anchor_curves": anchor_curves}

def save_models(models, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(models, path, compress=3)


def load_models(path):
    return joblib.load(path)

def calibrate(p_et, cfg, offset=None, floor=True):
    c = cfg["calibration"]
    p = np.clip(np.asarray(p_et, dtype=float), 1e-4, 1 - 1e-4)
    z = c["a"] + c["b"] * np.log(p / (1.0 - p))
    if offset is not None:
        z = z + offset
    P = 1.0 / (1.0 + np.exp(-z))
    return np.clip(P, c["p_floor"], 1 - 1e-4) if floor else P


def history_terms(keys, lag_train, cfg, anchor_pet=None, anchor_curves=None, curves=None):
    hc = cfg["history"]
    taus = np.asarray(hc["taus"], dtype=float)
    beta = np.asarray(hc["beta"], dtype=float)
    rhos = np.asarray(hc["rhos"], dtype=float)
    gamma = np.asarray(hc["gamma"], dtype=float)
    sim = hc.get("similarity", {})
    use_sim = bool(sim.get("enabled", False)) and anchor_curves is not None and curves is not None
    if use_sim:
        s_taus = np.asarray(sim["taus"], dtype=float); s_beta = np.asarray(sim["beta"], dtype=float)
        s_rhos = np.asarray(sim["rhos"], dtype=float); s_gamma = np.asarray(sim["gamma"], dtype=float)
        power = float(sim["power"])
        ak = anchor_curves["keys"]
        arow = {(g, i, pd.Timestamp(d)): j for j, (g, i, d) in
                enumerate(zip(ak.WELL_GROUP_NAME, ak.PROD_INDICATOR, ak.PROD_DATE))}
        AC = anchor_curves["curves"].astype(np.float64)
    vh = hc.get("value", {})
    use_val = bool(vh.get("enabled", False))
    if use_val:
        v_taus = np.asarray(vh["taus"], dtype=float); v_g = np.asarray(vh["g"], dtype=float)
    anchor_pet = anchor_pet or {}
    lab = lag_train[["WELL_GROUP_NAME", "PROD_INDICATOR", "PROD_DATE", "OPTIMAL_LAG_DAYS"]].sort_values("PROD_DATE")
    k = ["WELL_GROUP_NAME", "PROD_INDICATOR"]
    out = np.zeros(len(keys))
    vadd = np.zeros((len(keys), 4))
    hist = {key: (d.PROD_DATE.to_numpy(), d.OPTIMAL_LAG_DAYS.to_numpy()) for key, d in lab.groupby(k, sort=False)}
    kk = keys.reset_index(drop=True)
    for i, (g, ind, t) in enumerate(zip(kk.WELL_GROUP_NAME, kk.PROD_INDICATOR, kk.PROD_DATE)):
        h = hist.get((g, ind))
        if h is None:
            continue
        j = np.searchsorted(h[0], np.datetime64(t), side="left") - 1   # 严格早于 t 的最后一个标签
        if j < 0:
            continue
        ya = int(h[1][j])
        s = 1.0 if ya <= 3 else -1.0
        dd = (np.datetime64(t) - h[0][j]) / np.timedelta64(1, "D")
        akey = (g, ind, pd.Timestamp(h[0][j]))
        off = s * float((beta * np.exp(-dd / taus)).sum())
        pa = anchor_pet.get(akey)
        r = 0.0
        if pa is not None:
            r = (1.0 if ya <= 3 else 0.0) - float(calibrate(np.array([pa]), cfg, floor=False)[0])
            off += r * float((gamma * np.exp(-dd / rhos)).sum())
        if use_sim:
            ja = arow.get(akey)
            if ja is not None:
                c = float(_curve_corr(curves[i:i + 1], AC[ja:ja + 1])[0])
                c = max(0.0, min(1.0, c)) ** power
                off += c * (r * float((s_gamma * np.exp(-dd / s_rhos)).sum())
                            + s * float((s_beta * np.exp(-dd / s_taus)).sum()))
        out[i] = off
        if use_val and ya <= 3:
            vadd[i, ya] = float((v_g * np.exp(-dd / v_taus)).sum())
    return out, vadd


def emit(ps, p4, long_med, p_assume):
    w_short = 0.3 * 30.0 / p_assume
    w_long = 0.1 * 2.0 / (1.0 - p_assume)
    e_s = (p4[:, None, :] * np.abs(CAND[None, :, None] - SHORT_VALS[None, None, :])).sum(axis=2)
    e_l = np.abs(CAND[None, :] - long_med[:, None])
    cost = (ps[:, None] * (60.0 * (CAND > 3)[None, :] + w_short * e_s)
            + (1.0 - ps)[:, None] * (60.0 * (CAND <= 3)[None, :] + w_long * e_l))
    return CAND[np.argmin(cost, axis=1)].astype(int)


# ---------------- 推理 ----------------
def _detail(keys, X, P, p4, lm, pred):
    def col(name):
        return X[name].to_numpy(np.float64) if name in X.columns else np.full(len(X), np.nan)
    peak = col("pl.90.ma0.mxa")
    prod_obs = col("st.prod_obs")
    flag = np.full(len(X), "正常", dtype=object)
    flag[peak < 3] = "弱相关"
    flag[prod_obs < 0.5] = "采出观测不足"
    flag[np.isnan(peak)] = "滞后谱无定义"
    conf = np.maximum(P, 1.0 - P)
    return pd.DataFrame({
        "WELL_GROUP_NAME": keys.WELL_GROUP_NAME.to_numpy(),
        "PROD_DATE": keys.PROD_DATE.dt.strftime("%Y-%m-%d").to_numpy(),
        "PROD_INDICATOR": keys.PROD_INDICATOR.to_numpy(),
        "OPTIMAL_LAG_DAYS": pred,
        "P_SHORT": np.round(P, 4),
        "CONFIDENCE": np.round(conf, 4),
        "CONFIDENCE_LEVEL": np.where(conf >= 0.9, "高", np.where(conf >= 0.7, "中", "低")),
        "P_LAG0": np.round(p4[:, 0], 4), "P_LAG1": np.round(p4[:, 1], 4),
        "P_LAG2": np.round(p4[:, 2], 4), "P_LAG3": np.round(p4[:, 3], 4),
        "LONG_LAG_MEDIAN": np.round(lm, 1),
        "SPECTRUM_PEAK": np.round(peak, 2),
        "SPECTRUM_ARGMAX_LAG": col("pl.90.ma0.ama"),
        "INJ_ACTIVE_RATIO_90D": np.round(col("st.inj_nz"), 3),
        "PROD_OBS_RATIO_90D": np.round(prod_obs, 3),
        "DATA_FLAG": flag,
    })


def predict(models, data_dir, template, cfg, verbose=True, return_detail=False):
    keys = pd.DataFrame({
        "WELL_GROUP_NAME": [r["WELL_GROUP_NAME"] for r in template["rows"]],
        "PROD_DATE": pd.to_datetime([r["PROD_DATE"] for r in template["rows"]]),
        "PROD_INDICATOR": [r["PROD_INDICATOR"] for r in template["rows"]],
    })
    t0 = time.time()
    raw = load_data(data_dir, include_test=True)
    if verbose:
        print("[推理] 构造测试特征 (%d 行)" % len(keys), flush=True)
    X = build_features(SeriesBank(raw), keys, verbose=verbose, tag="测试", n_jobs=n_jobs_of(cfg))[models["columns"]]
    t1 = time.time()
    n = len(X)
    Xe = _et_matrix(X, cfg)
    p_et = np.mean([_p_short(et, Xe) for et in models["segment"]], axis=0)
    clf_s = models["short_value"]
    pv = clf_s.predict_proba(X)
    p4 = np.zeros((n, 4))
    for j, k in enumerate(clf_s.classes_):
        p4[:, int(k)] = pv[:, j]
    p4 /= np.maximum(p4.sum(axis=1, keepdims=True), 1e-12)
    lm = np.clip(models["long_value"].predict(X), 4, 45)
    off = None
    if cfg["history"]["enabled"]:
        curves = None
        if cfg["history"].get("similarity", {}).get("enabled", False):
            curves = np.nan_to_num(X[_curve_cols(cfg)].to_numpy(np.float64))
        off, vadd = history_terms(keys, raw["lag_train"], cfg, models.get("anchor_pet"),
                                  models.get("anchor_curves"), curves)
        if cfg["history"].get("value", {}).get("enabled", False):
            L = np.log(np.clip(p4, 1e-6, 1.0)) + vadd
            L -= L.max(axis=1, keepdims=True)
            p4 = np.exp(L)
            p4 /= p4.sum(axis=1, keepdims=True)
    P = calibrate(p_et, cfg, off)
    pred = emit(P, p4, lm, cfg["decision"]["p_assume"])
    if verbose:
        print("[推理] 特征构造用时 %.0f 秒，模型推理与决策用时 %.1f 秒" % (t1 - t0, time.time() - t1), flush=True)
        vc = pd.Series(pred).value_counts().sort_index()
        print("[推理] 判为短段比例 = %.4f；取值分布 = %s"
              % ((pred <= 3).mean(), {int(k): int(v) for k, v in vc.items()}), flush=True)
    if return_detail:
        return pred, _detail(keys, X, P, p4, lm, pred)
    return pred
