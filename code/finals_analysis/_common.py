import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_WORK = os.path.join(HERE, "outputs")


def use_code(code_dir=None):
    root = os.path.abspath(code_dir or PKG_ROOT)
    if not os.path.exists(os.path.join(root, "src", "feature.py")):
        raise SystemExit("找不到代码包：%s 下没有 src/feature.py" % root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def work_dir(path=None):
    d = os.path.abspath(path or DEFAULT_WORK)
    os.makedirs(d, exist_ok=True)
    return d


def quiet():
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import numpy as np
        np.seterr(all="ignore")
    except ImportError:
        pass
