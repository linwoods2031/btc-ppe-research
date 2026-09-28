import numpy as np
import pandas as pd

WINDOWS=(1,3,5,15)

def add_ppe_features(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy().sort_values("open_time").reset_index(drop=True)
    x["log_close"]=np.log(x["close"])
    # One-hour trailing quote-volume baseline; shifted so the current minute
    # cannot define its own normal-volume benchmark.
    baseline=x["quote_volume"].rolling(60,min_periods=30).mean().shift(1)
    for n in WINDOWS:
        ret=x["log_close"].diff(n)
        actual=x["quote_volume"].rolling(n,min_periods=n).sum()
        expected=baseline*n
        ppe=ret/(actual/expected.replace(0,np.nan))
        x[f"ppe_{n}"]=ppe.replace([np.inf,-np.inf],np.nan)
    p=x["ppe_5"]
    hist_mean=p.rolling(240,min_periods=120).mean().shift(1)
    hist_std=p.rolling(240,min_periods=120).std().shift(1)
    x["zppe_5"]=(p-hist_mean)/hist_std.replace(0,np.nan)
    x["ppe_slope"]=x["zppe_5"].diff()
    x["ppe_accel"]=x["ppe_slope"].diff()
    fast=x["zppe_5"].ewm(span=5,adjust=False).mean()
    slow=x["zppe_5"].ewm(span=20,adjust=False).mean()
    x["ppe_dif"]=fast-slow
    x["ppe_dea"]=x["ppe_dif"].ewm(span=9,adjust=False).mean()
    x["ppe_hist"]=x["ppe_dif"]-x["ppe_dea"]
    x["taker_imbalance"]=2*x["taker_buy_quote_volume"]/x["quote_volume"].replace(0,np.nan)-1
    x["fwd_ret_5m"]=x["log_close"].shift(-5)-x["log_close"]
    x["up_5m"]=(x["fwd_ret_5m"]>0).astype("Int64")
    x.loc[x["fwd_ret_5m"].isna(),"up_5m"]=pd.NA
    return x
