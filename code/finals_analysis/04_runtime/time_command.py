import os, sys, time, json, argparse, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _common as C

if "--" not in sys.argv:
    print(__doc__)
    sys.exit(0 if any(x in ("-h", "--help") for x in sys.argv[1:]) else 2)
k = sys.argv.index("--")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--name", default="run", help="本次测量的名字，用于输出文件名")
ap.add_argument("--work_dir", default=None, help="输出目录，缺省 finals_analysis/outputs/")
args = ap.parse_args(sys.argv[1:k]); cmd = sys.argv[k + 1:]
WORK = C.work_dir(args.work_dir)

t0 = time.time()
rc = subprocess.call(cmd)
sec = time.time() - t0
rec = dict(name=args.name, cmd=" ".join(cmd), returncode=rc, seconds=round(sec, 1))
try:
    import resource
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    rss = ru.ru_maxrss / (1024.0 * 1024.0) if sys.platform == "darwin" else ru.ru_maxrss / 1024.0
    rec.update(peak_rss_mb=round(rss, 1), cpu_seconds=round(ru.ru_utime + ru.ru_stime, 1))
except ImportError:
    pass
json.dump(rec, open(os.path.join(WORK, "timing_%s.json" % args.name), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("返回码 %d；用时 %.1f 秒（%.1f 分钟）%s" % (rc, sec, sec / 60.0,
      "；峰值内存 %.0f MB；CPU %.0f 秒" % (rec["peak_rss_mb"], rec["cpu_seconds"]) if "peak_rss_mb" in rec else ""))
sys.exit(rc)
