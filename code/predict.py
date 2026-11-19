from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.utils import load_config, read_submission_template, write_submission, resolve_output_path  # noqa: E402
from src import model as M

SEED = 20260916


def main(argv=None):
    ap = argparse.ArgumentParser(description="预测注采响应最佳滞后天数")
    ap.add_argument("--data_dir", required=True, help="数据目录（含 train/ 与 test/）")
    ap.add_argument("--output", required=True, help="结果文件路径或输出目录")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"), help="参数配置文件")
    ap.add_argument("--use_saved_model", action="store_true",
                    help="可选：跳过训练，直接加载 config 中 model_file 指定的已训练模型")
    ap.add_argument("--detail_output", default=None,
                    help="可选：另存一份逐样本诊断表（P(短段)、置信度、滞后谱峰值、数据质量标记等），供现场复核；不影响结果文件")
    ap.add_argument("--n_jobs", type=int, default=None,
                    help="可选：并行度，覆盖 config 中的 runtime.n_jobs（0 = 自动，1 = 单进程）；只影响速度，不影响结果")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    random.seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()
    verbose = not args.quiet
    cfg = load_config(args.config)
    if args.n_jobs is not None:
        cfg.setdefault("runtime", {})["n_jobs"] = args.n_jobs
    model_file = os.path.join(HERE, cfg["paths"]["model_file"])

    # 1) 训练、加载已训练模型
    if args.use_saved_model:
        if not os.path.exists(model_file):
            raise FileNotFoundError("找不到已训练的模型文件: %s。请先不带 --use_saved_model 运行一次，即可从 train/ 训练并生成它。" % model_file)
        models = M.load_models(model_file)
        if verbose:
            print("[加载] %s" % model_file, flush=True)
    else:
        models = M.train(args.data_dir, cfg, verbose=verbose)
        try:
            M.save_models(models, model_file)
            if verbose:
                print("[训练] 模型已保存: %s" % model_file, flush=True)
        except OSError as e:
            print("[训练] 模型未能保存（%s），继续推理" % e, flush=True)

    # 2) 推理：读取测试模板与测试期注采数据
    template_path = os.path.join(args.data_dir, "test", "test_optimal_lag_days.csv")
    if not os.path.exists(template_path):
        raise FileNotFoundError("找不到测试模板文件: %s" % template_path)
    template = read_submission_template(template_path)
    if args.detail_output:
        pred, detail = M.predict(models, args.data_dir, template, cfg, verbose=verbose, return_detail=True)
        d = os.path.dirname(os.path.abspath(args.detail_output))
        os.makedirs(d, exist_ok=True)
        detail.to_csv(args.detail_output, index=False, encoding="utf-8-sig")
        if verbose:
            print("[推理] 诊断表已写出: %s" % args.detail_output, flush=True)
    else:
        pred = M.predict(models, args.data_dir, template, cfg, verbose=verbose)

    # 3) 写出结果
    out_path = resolve_output_path(args.output)
    write_submission(template, pred, out_path)
    if verbose:
        print("已写出: %s  (%d 行, 用时 %.1f 秒)" % (out_path, len(pred), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
