import argparse, os, sys, tempfile, shutil, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _common as C

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data_dir", required=True, help="数据目录，其下为 train/ 与 test/")
ap.add_argument("--full_result", required=True, help="完整运行得到的结果 csv")
ap.add_argument("--code_dir", default=None, help="代码包根目录，缺省为本脚本所在代码包")
a = ap.parse_args()
code_dir = C.use_code(a.code_dir)

tpl = os.path.join(a.data_dir, "test", "test_optimal_lag_days.csv")
raw = open(tpl, "rb").read()
bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
nl = b"\r\n" if b"\r\n" in raw else b"\n"
lines = [l for l in raw[len(bom):].split(nl) if l != b""]
head, rows = lines[0], lines[1:]
sub = rows[::3][::-1]

tmp = tempfile.mkdtemp()
try:
    shutil.copytree(os.path.join(a.data_dir, "train"), os.path.join(tmp, "train"))
    os.makedirs(os.path.join(tmp, "test"))
    for f in os.listdir(os.path.join(a.data_dir, "test")):
        if f != "test_optimal_lag_days.csv":
            shutil.copy(os.path.join(a.data_dir, "test", f), os.path.join(tmp, "test", f))
    open(os.path.join(tmp, "test", "test_optimal_lag_days.csv"), "wb").write(bom + nl.join([head] + sub) + nl)
    out = os.path.join(tmp, "sub.csv")
    subprocess.check_call([sys.executable, os.path.join(code_dir, "predict.py"), "--data_dir", tmp, "--output", out,
                           "--use_saved_model", "--quiet"])
    key = lambda l: b",".join(l.split(b",")[:4])
    full = {}
    fr = open(a.full_result, "rb").read()
    fr = fr[3:] if fr.startswith(b"\xef\xbb\xbf") else fr
    for l in fr.replace(b"\r\n", b"\n").split(b"\n")[1:]:
        if l:
            full[key(l)] = l.split(b",")[4]
    got = [l for l in open(out, "rb").read()[len(bom):].split(nl)[1:] if l]
    assert len(got) == len(sub)
    diff = sum(1 for l in got if full[key(l)] != l.split(b",")[4])
    print("子集 %d 行（完整 %d 行），倒序推理；与完整运行结果不同的行数：%d" % (len(got), len(rows), diff))
    sys.exit(0 if diff == 0 else 1)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
