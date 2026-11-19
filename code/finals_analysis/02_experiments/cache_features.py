import argparse, os, sys, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _common as C

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data_dir", required=True, help="数据目录，其下为 train/ 与 test/")
ap.add_argument("--work_dir", default=None, help="输出目录，缺省 finals_analysis/outputs/")
ap.add_argument("--code_dir", default=None, help="代码包根目录，缺省为本脚本所在代码包")
a = ap.parse_args()
C.quiet(); C.use_code(a.code_dir)
DATA = a.data_dir; OUT = os.path.join(C.work_dir(a.work_dir), "feats.pkl")
import numpy as np, pandas as pd
from src import feature as F
from src.utils import read_submission_template
def build(bank, keys):
    T={}
    keys=keys.reset_index(drop=True)
    t=time.time(); A=F.build_frame(bank, keys).astype(np.float32); T['recipe_bank']=time.time()-t
    ti=keys.PROD_DATE.map(bank.pos).fillna(-1).astype(int).to_numpy()
    t=time.time(); R=F.build_lin_frame(bank.INJ, bank.PRO, keys.WELL_GROUP_NAME.to_numpy(), keys.PROD_INDICATOR.to_numpy(), ti); R.loc[ti<0,:]=np.nan; T['rank_spectra']=time.time()-t
    t=time.time(); S=F.shape_features(R); T['shape']=time.time()-t
    t=time.time(); W=F.window_stats(bank, keys); T['window_stats']=time.time()-t
    X=pd.concat([A.reset_index(drop=True),R.reset_index(drop=True),S,W],axis=1)
    return X,T
t0=time.time()
raw=F.load_data(DATA, include_test=False); lag=raw['lag_train']
t=time.time(); bank=F.SeriesBank(raw); tb=time.time()-t
Xtr,Ttr=build(bank, lag[["WELL_GROUP_NAME","PROD_DATE","PROD_INDICATOR"]])
print('train', Xtr.shape, 'bank %.1f'%tb, {k:round(v,1) for k,v in Ttr.items()}, flush=True)
tpl=read_submission_template(os.path.join(DATA,'test','test_optimal_lag_days.csv'))
keys=pd.DataFrame({"WELL_GROUP_NAME":[r["WELL_GROUP_NAME"] for r in tpl["rows"]],"PROD_DATE":pd.to_datetime([r["PROD_DATE"] for r in tpl["rows"]]),"PROD_INDICATOR":[r["PROD_INDICATOR"] for r in tpl["rows"]]})
raw2=F.load_data(DATA, include_test=True)
Xte,Tte=build(F.SeriesBank(raw2), keys)
print('test', Xte.shape, {k:round(v,1) for k,v in Tte.items()}, flush=True)
pickle.dump({'Xtr':Xtr,'lag':lag,'Xte':Xte,'keys':keys,'Ttr':Ttr,'Tte':Tte}, open(OUT,'wb'), protocol=4)
print('total %.1f s -> %s'%(time.time()-t0, OUT))
