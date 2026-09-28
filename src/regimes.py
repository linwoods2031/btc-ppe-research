import numpy as np
import pandas as pd

def add_regimes(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    utc=pd.to_datetime(x["open_time"],utc=True)
    ny=utc.dt.tz_convert("America/New_York")
    x["weekend"]=ny.dt.dayofweek>=5
    mins=ny.dt.hour*60+ny.dt.minute
    # Explicit session buckets. Weekend is kept as a separate first-class regime.
    x["ny_session"]=np.select(
        [x["weekend"], (mins>=480)&(mins<570), (mins>=570)&(mins<600),
         (mins>=600)&(mins<960), (mins>=960)],
        ["weekend","pre_market","open_0930_1000","us_cash","post_cash"],
        default="overnight")
    r=np.log(x["close"]).diff()
    x["ret_60m"]=np.log(x["close"]).diff(60)
    x["trend_regime"]=np.select(
        [x["ret_60m"]>r.rolling(1440,min_periods=240).std().shift(1),
         x["ret_60m"]<-r.rolling(1440,min_periods=240).std().shift(1)],
        ["up","down"],default="range")
    rv=r.rolling(60,min_periods=30).std()
    rv_ref=rv.rolling(1440,min_periods=240)
    x["vol_regime"]=np.select(
        [rv>rv_ref.quantile(.67).shift(1),rv<rv_ref.quantile(.33).shift(1)],
        ["high","low"],default="normal")
    q=x["quote_volume"]
    qref=q.rolling(1440,min_periods=240)
    x["liq_regime"]=np.select(
        [q>qref.quantile(.67).shift(1),q<qref.quantile(.33).shift(1)],
        ["high","low"],default="normal")
    return x
