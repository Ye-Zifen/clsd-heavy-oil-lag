"""
工具函数：配置读取、提交文件的字节级读写、评分函数
"""
from __future__ import annotations

import io
import os

import numpy as np
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def segment_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    gs = y_true <= 3
    acc = float(((y_pred <= 3) == gs).mean())
    mae_s = float(np.abs(y_pred[gs] - y_true[gs]).mean()) if gs.any() else 0.0
    mae_l = float(np.abs(y_pred[~gs] - y_true[~gs]).mean()) if (~gs).any() else 0.0
    s = 60.0 * acc + 0.3 * max(0.0, 100.0 - 30.0 * mae_s) + 0.1 * max(0.0, 100.0 - 2.0 * mae_l)
    return dict(acc=acc, mae_short=mae_s, mae_long=mae_l, score=s)


EXPECTED_HEADER = [
    "WELL_GROUP_NAME",
    "PROD_DATE",
    "INJ_INDICATOR",
    "PROD_INDICATOR",
    "OPTIMAL_LAG_DAYS",
]


def read_submission_template(path: str):
    with open(path, "rb") as f:
        raw = f.read()
    bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
    if bom:
        raw = raw[len(bom):]
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8")
    lines = text.split(newline)
    trailing_newline = lines and lines[-1] == ""
    if trailing_newline:
        lines = lines[:-1]
    header_line = lines[0]
    header = [c.strip() for c in header_line.split(",")]
    if header != EXPECTED_HEADER:
        raise ValueError("提交模板列名与规范不一致: %r" % (header,))

    rows = []
    for ln in lines[1:]:
        if ln == "":
            continue
        parts = ln.split(",")
        prefix = ",".join(parts[:4])
        rows.append(
            {
                "prefix": prefix,
                "WELL_GROUP_NAME": parts[0],
                "PROD_DATE": parts[1],
                "INJ_INDICATOR": parts[2],
                "PROD_INDICATOR": parts[3],
            }
        )
    return {
        "bom": bom,
        "newline": newline,
        "header_line": header_line,
        "rows": rows,
        "trailing_newline": trailing_newline,
    }


def write_submission(template: dict, values, path: str) -> None:
    rows = template["rows"]
    if len(values) != len(rows):
        raise ValueError("预测数量 %d 与模板行数 %d 不一致" % (len(values), len(rows)))
    nl = template["newline"]
    out = io.StringIO()
    out.write(template["header_line"])
    out.write(nl)
    for r, v in zip(rows, values):
        iv = int(v)
        if iv < 0 or iv > 45:
            raise ValueError("预测值 %r 超出 [0,45]" % (v,))
        out.write(r["prefix"])
        out.write(",")
        out.write(str(iv))
        out.write(nl)
    data = template["bom"] + out.getvalue().encode("utf-8")
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def resolve_output_path(output: str) -> str:
    if output.lower().endswith(".csv"):
        return output
    return os.path.join(output, "test_optimal_lag_days.csv")
