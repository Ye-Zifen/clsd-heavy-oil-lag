import os, sys, json, pickle, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import _common as C

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("pages", nargs="*", help="只画这些")
ap.add_argument("--work_dir", default=None, help="01–03 与 prepare_data.py 的输出目录，缺省 finals_analysis/outputs/")
ap.add_argument("--code_dir", default=None, help="代码包根目录，缺省为本脚本所在代码包")
a = ap.parse_args()
C.quiet(); CODE = C.use_code(a.code_dir); WORK = C.work_dir(a.work_dir)
from src.utils import load_config
import figs_1_problem, figs_2_method, figs_3_experiments, figs_4_engineering, figs_5_application

P = pickle.load(open(os.path.join(WORK, "figure_data.pkl"), "rb"))
E = json.load(open(os.path.join(WORK, "numbers_exp.json"), encoding="utf-8"))
R = json.load(open(os.path.join(HERE, "recorded_numbers.json"), encoding="utf-8"))
cfg = load_config(os.path.join(CODE, "config.yaml"))
out = os.path.join(WORK, "figures")

fns = (figs_1_problem.FIGS + figs_2_method.FIGS + figs_3_experiments.FIGS + figs_4_engineering.FIGS + figs_5_application.FIGS)
for fn in fns:
    if a.pages and not any(fn.__name__.startswith(p) for p in a.pages):
        continue
    print(fn(P, R, E, cfg, out))
